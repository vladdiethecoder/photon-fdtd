//! CUDA backend: the same flat f64 arrays and the same update, one thread per cell.
//!
//! The kernels are compiled with `--fmad=false` and keep the CPU operation
//! order, so CPU and GPU results are bitwise identical.

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::{Yee2d, cell_coeffs, check_setup};

const KERNELS: &str = r#"
extern "C" __global__ void update_h(double* __restrict__ hx, double* __restrict__ hy,
                                    const double* __restrict__ ez, const double* __restrict__ ca,
                                    const double* __restrict__ hb, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx - 1) return;
    // Grid-stride over rows: gridDim.y is capped at 65535 by CUDA.
    for (int j = blockIdx.y; j < ny - 1; j += gridDim.y) {
        size_t k = (size_t)j * (size_t)nx + (size_t)i;
        double e = ez[k];
        double a = ca[k];
        double b = hb[k];
        hx[k] = a * hx[k] - b * (ez[k + nx] - e);
        hy[k] = a * hy[k] + b * (ez[k + 1] - e);
    }
}

extern "C" __global__ void update_e(double* __restrict__ ez, const double* __restrict__ hx,
                                    const double* __restrict__ hy, const double* __restrict__ ca,
                                    const double* __restrict__ cb, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= nx - 1) return;
    for (int j = 1 + blockIdx.y; j < ny - 1; j += gridDim.y) {
        size_t k = (size_t)j * (size_t)nx + (size_t)i;
        ez[k] = ca[k] * ez[k] + cb[k] * ((hy[k] - hy[k - 1]) - (hx[k] - hx[k - nx]));
    }
}

extern "C" __global__ void add_source_line(double* ez, int nx, int i, int j0, int j1, double amp) {
    int j = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= j1) return;
    size_t k = (size_t)j * (size_t)nx + (size_t)i;
    ez[k] = ez[k] + amp;
}
"#;

const BLOCK_X: u32 = 256;
const MAX_GRID_Y: u32 = 65535;

pub struct GpuYee {
    pub nx: usize,
    pub ny: usize,
    stream: Arc<CudaStream>,
    update_h: CudaFunction,
    update_e: CudaFunction,
    add_src: CudaFunction,
    ez: CudaSlice<f64>,
    hx: CudaSlice<f64>,
    hy: CudaSlice<f64>,
    ca: CudaSlice<f64>,
    cb: CudaSlice<f64>,
    hb: CudaSlice<f64>,
}

impl GpuYee {
    fn functions(stream: &Arc<CudaStream>) -> Result<[CudaFunction; 3], Box<dyn Error>> {
        let ptx = compile_ptx_with_opts(
            KERNELS,
            CompileOptions { fmad: Some(false), arch: Some("sm_120"), ..Default::default() },
        )?;
        let module = stream.context().load_module(ptx)?;
        Ok([
            module.load_function("update_h")?,
            module.load_function("update_e")?,
            module.load_function("add_source_line")?,
        ])
    }

    /// Upload the fields and coefficients of a CPU solver.
    pub fn from_cpu(sim: &Yee2d) -> Result<Self, Box<dyn Error>> {
        let stream = CudaContext::new(0)?.default_stream();
        let [update_h, update_e, add_src] = Self::functions(&stream)?;
        Ok(Self {
            nx: sim.nx,
            ny: sim.ny,
            update_h,
            update_e,
            add_src,
            ez: stream.clone_htod(&sim.ez)?,
            hx: stream.clone_htod(&sim.hx)?,
            hy: stream.clone_htod(&sim.hy)?,
            ca: stream.clone_htod(&sim.ca)?,
            cb: stream.clone_htod(&sim.cb)?,
            hb: stream.clone_htod(&sim.hb)?,
            stream,
        })
    }

    /// Build directly on the device, without a full-size host copy: fields
    /// start at zero, coefficients are computed on the host in row blocks
    /// with the same `cell_coeffs` as the CPU solver. Use this for grids that
    /// do not fit in host RAM.
    pub fn new(
        nx: usize,
        ny: usize,
        sponge: usize,
        courant: f64,
        eps_r: &dyn Fn(usize, usize) -> f64,
    ) -> Result<Self, Box<dyn Error>> {
        check_setup(nx, ny, courant)?;
        if nx > i32::MAX as usize || ny > i32::MAX as usize {
            return Err("nx and ny must fit in i32".into());
        }
        let n = nx * ny;
        let stream = CudaContext::new(0)?.default_stream();
        let [update_h, update_e, add_src] = Self::functions(&stream)?;
        let (mut ca, mut cb, mut hb) =
            (stream.alloc_zeros::<f64>(n)?, stream.alloc_zeros::<f64>(n)?, stream.alloc_zeros::<f64>(n)?);
        let rows = (1usize << 22).div_ceil(nx).max(1); // ~32 MiB per host block
        let (mut ba, mut bb, mut bh) = (Vec::new(), Vec::new(), Vec::new());
        for j0 in (0..ny).step_by(rows) {
            let j1 = (j0 + rows).min(ny);
            ba.clear();
            bb.clear();
            bh.clear();
            for j in j0..j1 {
                for i in 0..nx {
                    let (a, b, h) = cell_coeffs(i, j, nx, ny, sponge, courant, eps_r(i, j));
                    ba.push(a);
                    bb.push(b);
                    bh.push(h);
                }
            }
            let r = j0 * nx..j1 * nx;
            stream.memcpy_htod(&ba, &mut ca.try_slice_mut(r.clone()).unwrap())?;
            stream.memcpy_htod(&bb, &mut cb.try_slice_mut(r.clone()).unwrap())?;
            stream.memcpy_htod(&bh, &mut hb.try_slice_mut(r).unwrap())?;
            stream.synchronize()?; // host blocks are reused
        }
        Ok(Self {
            nx,
            ny,
            update_h,
            update_e,
            add_src,
            ez: stream.alloc_zeros::<f64>(n)?,
            hx: stream.alloc_zeros::<f64>(n)?,
            hy: stream.alloc_zeros::<f64>(n)?,
            ca,
            cb,
            hb,
            stream,
        })
    }

    /// Overwrite Ez at flat index `k`.
    pub fn set_ez(&mut self, k: usize, value: f64) -> Result<(), Box<dyn Error>> {
        let mut dst = self.ez.try_slice_mut(k..k + 1).ok_or("index out of range")?;
        self.stream.memcpy_htod(&[value], &mut dst)?;
        Ok(())
    }

    /// Download Ez[range] (flat indices).
    pub fn download_ez_range(&self, r: std::ops::Range<usize>) -> Result<Vec<f64>, Box<dyn Error>> {
        Ok(self.stream.clone_dtoh(&self.ez.try_slice(r).ok_or("range out of bounds")?)?)
    }

    pub fn device_name(&self) -> String {
        self.stream.context().name().unwrap_or_else(|_| "unknown".into())
    }

    fn cfg(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((self.nx as u32).div_ceil(BLOCK_X), (self.ny as u32).min(MAX_GRID_Y), 1),
            block_dim: (BLOCK_X, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Enqueue one Yee step (asynchronous).
    pub fn step(&mut self) -> Result<(), Box<dyn Error>> {
        let cfg = self.cfg();
        let (nx, ny) = (self.nx as i32, self.ny as i32);
        unsafe {
            self.stream
                .launch_builder(&self.update_h)
                .arg(&mut self.hx)
                .arg(&mut self.hy)
                .arg(&self.ez)
                .arg(&self.ca)
                .arg(&self.hb)
                .arg(&nx)
                .arg(&ny)
                .launch(cfg)?;
            self.stream
                .launch_builder(&self.update_e)
                .arg(&mut self.ez)
                .arg(&self.hx)
                .arg(&self.hy)
                .arg(&self.ca)
                .arg(&self.cb)
                .arg(&nx)
                .arg(&ny)
                .launch(cfg)?;
        }
        Ok(())
    }

    pub fn add_source_line(&mut self, i: usize, j0: usize, j1: usize, amp: f64) -> Result<(), Box<dyn Error>> {
        let n = (j1 - j0) as u32;
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (nx, i, j0, j1) = (self.nx as i32, i as i32, j0 as i32, j1 as i32);
        unsafe {
            self.stream
                .launch_builder(&self.add_src)
                .arg(&mut self.ez)
                .arg(&nx)
                .arg(&i)
                .arg(&j0)
                .arg(&j1)
                .arg(&amp)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Run `steps` Yee steps and return the GPU time in milliseconds (CUDA events).
    pub fn timed_steps(&mut self, steps: usize) -> Result<f64, Box<dyn Error>> {
        let start = self.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..steps {
            self.step()?;
        }
        let end = self.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        Ok(start.elapsed_ms(&end)? as f64)
    }

    pub fn synchronize(&self) -> Result<(), Box<dyn Error>> {
        Ok(self.stream.synchronize()?)
    }

    pub fn download_ez(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        Ok(self.stream.clone_dtoh(&self.ez)?)
    }
}
