//! CUDA backend: the same flat f64 arrays and the same update, one thread per cell.
//!
//! The kernels are compiled with `--fmad=false` and keep the CPU operation
//! order, so CPU and GPU results are bitwise identical. The CPML tables and
//! the strip geometry come from the same `CpmlLayout` as the CPU solver.

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::cpml::Profile;
use crate::{CpmlLayout, Pml, Yee2d, cb_coeff, check_setup};

const KERNELS: &str = r#"
#define K(i, j) ((size_t)(j) * (size_t)nx + (size_t)(i))

// Main H update. Grid-stride over rows: gridDim.y is capped at 65535 by CUDA.
extern "C" __global__ void update_h(double* __restrict__ hx, double* __restrict__ hy,
                                    const double* __restrict__ ez,
                                    const double* __restrict__ ikx_h, const double* __restrict__ iky_h,
                                    double dt, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx - 1) return;
    double ikx = ikx_h[i];
    for (int j = blockIdx.y; j < ny - 1; j += gridDim.y) {
        size_t k = K(i, j);
        double e = ez[k];
        hx[k] = hx[k] - dt * ((ez[k + nx] - e) * iky_h[j]);
        hy[k] = hy[k] + dt * ((ez[k + 1] - e) * ikx);
    }
}

// x strips, Hy: psi = b psi + a dEz/dx ; Hy += dt psi.  c in 0..2w.
extern "C" __global__ void psi_h_x(double* __restrict__ hy, double* __restrict__ psi,
                                   const double* __restrict__ ez,
                                   const double* __restrict__ a, const double* __restrict__ b,
                                   double dt, int nx, int ny, int w) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 2 * w) return;
    int i = c < w ? c : nx - 2 * w + c;
    if (i >= nx - 1) return;
    for (int j = blockIdx.y; j < ny - 1; j += gridDim.y) {
        size_t k = K(i, j);
        size_t s = (size_t)j * (size_t)(2 * w) + (size_t)c;
        double p = b[i] * psi[s] + a[i] * (ez[k + 1] - ez[k]);
        psi[s] = p;
        hy[k] = hy[k] + dt * p;
    }
}

// y strips, Hx: psi = b psi + a dEz/dy ; Hx -= dt psi.  r = blockIdx.y in 0..2w.
extern "C" __global__ void psi_h_y(double* __restrict__ hx, double* __restrict__ psi,
                                   const double* __restrict__ ez,
                                   const double* __restrict__ a, const double* __restrict__ b,
                                   double dt, int nx, int ny, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int j = r < w ? r : ny - 2 * w + r;
    if (i >= nx - 1 || j >= ny - 1) return;
    size_t k = K(i, j);
    size_t s = (size_t)r * (size_t)nx + (size_t)i;
    double p = b[j] * psi[s] + a[j] * (ez[k + nx] - ez[k]);
    psi[s] = p;
    hx[k] = hx[k] - dt * p;
}

// Main Ez update.
extern "C" __global__ void update_e(double* __restrict__ ez, const double* __restrict__ hx,
                                    const double* __restrict__ hy, const double* __restrict__ cb,
                                    const double* __restrict__ ikx_e, const double* __restrict__ iky_e,
                                    int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= nx - 1) return;
    double ikx = ikx_e[i];
    for (int j = 1 + blockIdx.y; j < ny - 1; j += gridDim.y) {
        size_t k = K(i, j);
        ez[k] = ez[k] + cb[k] * ((hy[k] - hy[k - 1]) * ikx - (hx[k] - hx[k - nx]) * iky_e[j]);
    }
}

// x strips, Ez: psi = b psi + a dHy/dx ; Ez += cb psi.
extern "C" __global__ void psi_e_x(double* __restrict__ ez, double* __restrict__ psi,
                                   const double* __restrict__ hy, const double* __restrict__ cb,
                                   const double* __restrict__ a, const double* __restrict__ b,
                                   int nx, int ny, int w) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 2 * w) return;
    int i = c < w ? c : nx - 2 * w + c;
    if (i < 1 || i >= nx - 1) return;
    for (int j = 1 + blockIdx.y; j < ny - 1; j += gridDim.y) {
        size_t k = K(i, j);
        size_t s = (size_t)j * (size_t)(2 * w) + (size_t)c;
        double p = b[i] * psi[s] + a[i] * (hy[k] - hy[k - 1]);
        psi[s] = p;
        ez[k] = ez[k] + cb[k] * p;
    }
}

// y strips, Ez: psi = b psi + a dHx/dy ; Ez -= cb psi.
extern "C" __global__ void psi_e_y(double* __restrict__ ez, double* __restrict__ psi,
                                   const double* __restrict__ hx, const double* __restrict__ cb,
                                   const double* __restrict__ a, const double* __restrict__ b,
                                   int nx, int ny, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int j = r < w ? r : ny - 2 * w + r;
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) return;
    size_t k = K(i, j);
    size_t s = (size_t)r * (size_t)nx + (size_t)i;
    double p = b[j] * psi[s] + a[j] * (hx[k] - hx[k - nx]);
    psi[s] = p;
    ez[k] = ez[k] - cb[k] * p;
}

// Set cb (= dt / eps_r) to `value` inside the disk |(i, j) - (cx, cy)| <= r.
extern "C" __global__ void paint_disk(double* cb, int nx, int ny, int cx, int cy, int r, double value) {
    int i = cx - r + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = cy - r + (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    int dx = i - cx, dy = j - cy;
    if (dx * dx + dy * dy > r * r) return;
    cb[K(i, j)] = value;
}

// Ez heat map into RGBA8 (0xAABBGGRR), row 0 of the image = top row of the grid.
// Red = +Ez, blue = -Ez, intensity sqrt(|gain Ez|) clipped to 1. Dielectric
// cells (cb < dt) get a grey base, the CPML strips a dark base.
extern "C" __global__ void render_rgba(const double* __restrict__ ez, const double* __restrict__ cb,
                                       unsigned long long out_ptr, int nx, int ny, double gain,
                                       double dt, int pml) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    unsigned int* out = (unsigned int*)out_ptr;
    size_t k = K(i, j);
    double v = ez[k] * gain;
    double a = sqrt(fmin(fabs(v), 1.0));
    double eps = dt / cb[k];
    int base = eps > 1.0 ? (int)(28.0 + 6.0 * fmin(eps, 13.0)) : 0;
    int edge = min(min(i, nx - 1 - i), min(j, ny - 1 - j));
    if (edge < pml) base = max(base, 18);
    int c = (int)(a * 255.0);
    int r = v > 0.0 ? max(c, base) : base;
    int b = v < 0.0 ? max(c, base) : base;
    int g = base;
    out[(size_t)(ny - 1 - j) * (size_t)nx + (size_t)i] = 0xFF000000u | ((unsigned)b << 16) | ((unsigned)g << 8) | (unsigned)r;
}

extern "C" __global__ void add_source_line(double* ez, int nx, int i, int j0, int j1, double amp) {
    int j = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= j1) return;
    size_t k = K(i, j);
    ez[k] = ez[k] + amp;
}
"#;

const BLOCK_X: u32 = 256;
const STRIP_BLOCK_X: u32 = 32;
const MAX_GRID_Y: u32 = 65535;

const NAMES: [&str; 9] =
    ["update_h", "psi_h_x", "psi_h_y", "update_e", "psi_e_x", "psi_e_y", "add_source_line", "paint_disk", "render_rgba"];

struct DevProfile {
    ik: CudaSlice<f64>,
    a: CudaSlice<f64>,
    b: CudaSlice<f64>,
}

impl DevProfile {
    fn upload(stream: &Arc<CudaStream>, p: &Profile) -> Result<Self, Box<dyn Error>> {
        Ok(Self { ik: stream.clone_htod(&p.ik)?, a: stream.clone_htod(&p.a)?, b: stream.clone_htod(&p.b)? })
    }
}

pub struct GpuYee {
    pub nx: usize,
    pub ny: usize,
    pub courant: f64,
    pub layout: CpmlLayout,
    stream: Arc<CudaStream>,
    f: [CudaFunction; 9],
    ez: CudaSlice<f64>,
    hx: CudaSlice<f64>,
    hy: CudaSlice<f64>,
    cb: CudaSlice<f64>,
    x_int: DevProfile,
    x_half: DevProfile,
    y_int: DevProfile,
    y_half: DevProfile,
    psi_ezx: CudaSlice<f64>,
    psi_hyx: CudaSlice<f64>,
    psi_ezy: CudaSlice<f64>,
    psi_hxy: CudaSlice<f64>,
}

impl GpuYee {
    fn functions(stream: &Arc<CudaStream>) -> Result<[CudaFunction; 9], Box<dyn Error>> {
        let ptx = compile_ptx_with_opts(
            KERNELS,
            CompileOptions { fmad: Some(false), arch: Some("sm_120"), ..Default::default() },
        )?;
        let module = stream.context().load_module(ptx)?;
        let mut out = Vec::with_capacity(NAMES.len());
        for n in NAMES {
            out.push(module.load_function(n)?);
        }
        Ok(out.try_into().map_err(|_| "kernel count")?)
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        stream: Arc<CudaStream>,
        layout: CpmlLayout,
        courant: f64,
        ez: CudaSlice<f64>,
        hx: CudaSlice<f64>,
        hy: CudaSlice<f64>,
        cb: CudaSlice<f64>,
        psi: [CudaSlice<f64>; 4],
    ) -> Result<Self, Box<dyn Error>> {
        if layout.nx > i32::MAX as usize || layout.ny > i32::MAX as usize {
            return Err("nx and ny must fit in i32".into());
        }
        if 2 * layout.w > MAX_GRID_Y as usize {
            return Err("PML too thick for one y-strip launch".into());
        }
        let [psi_ezx, psi_hyx, psi_ezy, psi_hxy] = psi;
        Ok(Self {
            nx: layout.nx,
            ny: layout.ny,
            courant,
            f: Self::functions(&stream)?,
            x_int: DevProfile::upload(&stream, &layout.x_int)?,
            x_half: DevProfile::upload(&stream, &layout.x_half)?,
            y_int: DevProfile::upload(&stream, &layout.y_int)?,
            y_half: DevProfile::upload(&stream, &layout.y_half)?,
            layout,
            ez,
            hx,
            hy,
            cb,
            psi_ezx,
            psi_hyx,
            psi_ezy,
            psi_hxy,
            stream,
        })
    }

    /// Upload the fields, coefficients, CPML layout and psi arrays of a CPU solver.
    pub fn from_cpu(sim: &Yee2d) -> Result<Self, Box<dyn Error>> {
        let stream = CudaContext::new(0)?.default_stream();
        let psi = [
            stream.clone_htod(&sim.psi_ezx)?,
            stream.clone_htod(&sim.psi_hyx)?,
            stream.clone_htod(&sim.psi_ezy)?,
            stream.clone_htod(&sim.psi_hxy)?,
        ];
        Self::assemble(
            stream.clone(),
            sim.layout.clone(),
            sim.courant,
            stream.clone_htod(&sim.ez)?,
            stream.clone_htod(&sim.hx)?,
            stream.clone_htod(&sim.hy)?,
            stream.clone_htod(&sim.cb)?,
            psi,
        )
    }

    /// Build directly on the device, without a full-size host copy: fields
    /// start at zero, and `cb` is computed on the host in row blocks with the
    /// same `cb_coeff` as the CPU solver. Use this for grids that do not fit
    /// in host RAM.
    pub fn new(
        nx: usize,
        ny: usize,
        pml: Pml,
        courant: f64,
        eps_r: &dyn Fn(usize, usize) -> f64,
    ) -> Result<Self, Box<dyn Error>> {
        check_setup(nx, ny, courant)?;
        let layout = CpmlLayout::new(nx, ny, pml, courant)?;
        let n = nx * ny;
        let stream = CudaContext::new(0)?.default_stream();
        let mut cb = stream.alloc_zeros::<f64>(n)?;
        let rows = (1usize << 22).div_ceil(nx).max(1); // ~32 MiB per host block
        let mut block = Vec::new();
        for j0 in (0..ny).step_by(rows) {
            let j1 = (j0 + rows).min(ny);
            block.clear();
            for j in j0..j1 {
                for i in 0..nx {
                    let e = eps_r(i, j);
                    if !(e >= 1.0) {
                        return Err("eps_r must be >= 1".into());
                    }
                    block.push(cb_coeff(courant, e));
                }
            }
            stream.memcpy_htod(&block, &mut cb.try_slice_mut(j0 * nx..j1 * nx).unwrap())?;
            stream.synchronize()?; // the host block is reused
        }
        let psi = [
            stream.alloc_zeros::<f64>(layout.x_strip_len())?,
            stream.alloc_zeros::<f64>(layout.x_strip_len())?,
            stream.alloc_zeros::<f64>(layout.y_strip_len())?,
            stream.alloc_zeros::<f64>(layout.y_strip_len())?,
        ];
        let (ez, hx, hy) = (stream.alloc_zeros::<f64>(n)?, stream.alloc_zeros::<f64>(n)?, stream.alloc_zeros::<f64>(n)?);
        Self::assemble(stream, layout, courant, ez, hx, hy, cb, psi)
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

    /// Enqueue one Yee step (asynchronous): main H, H psi (x, y), main E, E psi (x, y).
    pub fn step(&mut self) -> Result<(), Box<dyn Error>> {
        let (nx, ny, w) = (self.nx as i32, self.ny as i32, self.layout.w as i32);
        let dt = self.courant;
        let main = LaunchConfig {
            grid_dim: ((self.nx as u32).div_ceil(BLOCK_X), (self.ny as u32).min(MAX_GRID_Y), 1),
            block_dim: (BLOCK_X, 1, 1),
            shared_mem_bytes: 0,
        };
        let xs = LaunchConfig {
            grid_dim: ((2 * w as u32).div_ceil(STRIP_BLOCK_X), (self.ny as u32).min(MAX_GRID_Y), 1),
            block_dim: (STRIP_BLOCK_X, 1, 1),
            shared_mem_bytes: 0,
        };
        let ys = LaunchConfig {
            grid_dim: ((self.nx as u32).div_ceil(BLOCK_X), 2 * w as u32, 1),
            block_dim: (BLOCK_X, 1, 1),
            shared_mem_bytes: 0,
        };
        let s = &self.stream;
        unsafe {
            s.launch_builder(&self.f[0])
                .arg(&mut self.hx)
                .arg(&mut self.hy)
                .arg(&self.ez)
                .arg(&self.x_half.ik)
                .arg(&self.y_half.ik)
                .arg(&dt)
                .arg(&nx)
                .arg(&ny)
                .launch(main)?;
            s.launch_builder(&self.f[1])
                .arg(&mut self.hy)
                .arg(&mut self.psi_hyx)
                .arg(&self.ez)
                .arg(&self.x_half.a)
                .arg(&self.x_half.b)
                .arg(&dt)
                .arg(&nx)
                .arg(&ny)
                .arg(&w)
                .launch(xs)?;
            s.launch_builder(&self.f[2])
                .arg(&mut self.hx)
                .arg(&mut self.psi_hxy)
                .arg(&self.ez)
                .arg(&self.y_half.a)
                .arg(&self.y_half.b)
                .arg(&dt)
                .arg(&nx)
                .arg(&ny)
                .arg(&w)
                .launch(ys)?;
            s.launch_builder(&self.f[3])
                .arg(&mut self.ez)
                .arg(&self.hx)
                .arg(&self.hy)
                .arg(&self.cb)
                .arg(&self.x_int.ik)
                .arg(&self.y_int.ik)
                .arg(&nx)
                .arg(&ny)
                .launch(main)?;
            s.launch_builder(&self.f[4])
                .arg(&mut self.ez)
                .arg(&mut self.psi_ezx)
                .arg(&self.hy)
                .arg(&self.cb)
                .arg(&self.x_int.a)
                .arg(&self.x_int.b)
                .arg(&nx)
                .arg(&ny)
                .arg(&w)
                .launch(xs)?;
            s.launch_builder(&self.f[5])
                .arg(&mut self.ez)
                .arg(&mut self.psi_ezy)
                .arg(&self.hx)
                .arg(&self.cb)
                .arg(&self.y_int.a)
                .arg(&self.y_int.b)
                .arg(&nx)
                .arg(&ny)
                .arg(&w)
                .launch(ys)?;
        }
        Ok(())
    }

    pub fn add_source_line(&mut self, i: usize, j0: usize, j1: usize, amp: f64) -> Result<(), Box<dyn Error>> {
        let n = (j1 - j0) as u32;
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (nx, i, j0, j1) = (self.nx as i32, i as i32, j0 as i32, j1 as i32);
        unsafe {
            self.stream
                .launch_builder(&self.f[6])
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

    /// Set the permittivity of every cell inside a disk. `cb` is computed on the
    /// host with `cb_coeff`, so a painted cell holds the same bits as a cell
    /// built by the constructors.
    pub fn paint_disk(&mut self, cx: usize, cy: usize, r: usize, eps: f64) -> Result<(), Box<dyn Error>> {
        if !(eps >= 1.0) {
            return Err("eps_r must be >= 1".into());
        }
        let value = cb_coeff(self.courant, eps);
        let side = (2 * r + 1) as u32;
        let cfg = LaunchConfig { grid_dim: (side.div_ceil(16), side.div_ceil(16), 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 };
        let (nx, ny, cx, cy, r) = (self.nx as i32, self.ny as i32, cx as i32, cy as i32, r as i32);
        unsafe {
            self.stream
                .launch_builder(&self.f[7])
                .arg(&mut self.cb)
                .arg(&nx)
                .arg(&ny)
                .arg(&cx)
                .arg(&cy)
                .arg(&r)
                .arg(&value)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Write the RGBA8 heat map of Ez (nx * ny u32 values) to device address
    /// `out`, for example a mapped OpenGL buffer.
    ///
    /// # Safety
    /// `out` must point to at least nx * ny writable u32 values on this device.
    pub unsafe fn render_rgba_to_ptr(&self, out: u64, gain: f64) -> Result<(), Box<dyn Error>> {
        let cfg = LaunchConfig {
            grid_dim: ((self.nx as u32).div_ceil(32), (self.ny as u32).div_ceil(8), 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let (nx, ny, pml) = (self.nx as i32, self.ny as i32, self.layout.pml.cells as i32);
        let dt = self.courant;
        unsafe {
            self.stream
                .launch_builder(&self.f[8])
                .arg(&self.ez)
                .arg(&self.cb)
                .arg(&out)
                .arg(&nx)
                .arg(&ny)
                .arg(&gain)
                .arg(&dt)
                .arg(&pml)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Write the RGBA8 heat map of Ez into a device buffer of nx * ny u32 values.
    pub fn render_rgba(&self, out: &mut CudaSlice<u32>, gain: f64) -> Result<(), Box<dyn Error>> {
        if out.len() < self.nx * self.ny {
            return Err("image buffer too small".into());
        }
        let (ptr, _sync) = out.device_ptr_mut(&self.stream);
        unsafe { self.render_rgba_to_ptr(ptr, gain) }
    }

    /// Set all fields and psi arrays to zero. The geometry (`cb`) stays.
    pub fn reset_fields(&mut self) -> Result<(), Box<dyn Error>> {
        for a in [&mut self.ez, &mut self.hx, &mut self.hy, &mut self.psi_ezx, &mut self.psi_hyx, &mut self.psi_ezy, &mut self.psi_hxy] {
            self.stream.memset_zeros(a)?;
        }
        Ok(())
    }

    /// The CUDA stream of this solver (for interop with other APIs).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Download cb (dt / eps_r per cell).
    pub fn download_cb(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        Ok(self.stream.clone_dtoh(&self.cb)?)
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
