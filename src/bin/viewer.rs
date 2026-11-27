//! Interactive viewer: `cargo run --release --features gui --bin viewer`.
//!
//! The solver runs on the GPU. Each frame, a CUDA kernel writes the Ez heat map
//! and the eps_r overlay into an OpenGL pixel buffer that CUDA has mapped
//! (cuGraphicsGLRegisterBuffer). OpenGL then copies the buffer into the texture
//! that egui shows. The field data never leaves the GPU. If CUDA cannot register
//! the buffer (for example, if the GL context runs on another GPU), the viewer
//! uses a host copy of the 4-byte-per-cell image instead and shows that path.
//!
//! Controls: run / pause / single step, a dielectric brush (paint silicon,
//! erase to air), click to place the source, and a gain slider.
//!
//! Options:
//!   --host-copy            use the host-copy path even if interop works
//!   --capture FILE.ppm     after --steps N steps, save a screenshot and exit
//!   --steps N              (default 3000)
//!   --script FILE          replay pointer input through egui (see `Cmd`)
//!   F12                    save a screenshot to out/viewer_f12.ppm at any time

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::{self, CUgraphicsResource, CUresult};
use eframe::egui;
use eframe::glow::{self, HasContext};

use photon_fdtd::Pml;
use photon_fdtd::gpu::GpuYee;
use photon_fdtd::verify::{EPS_AIR, EPS_SI, courant};

const NX: usize = 1024;
const NY: usize = 512;
const PML: usize = 10;
const CORE_W: usize = 16;
const LAMBDA0: f64 = 40.0;

/// Initial geometry: a silicon strip from the left edge that bends 90 degrees
/// up to the top edge.
fn initial_eps(i: usize, j: usize) -> f64 {
    let (y0, y1) = (NY / 2 - CORE_W / 2, NY / 2 + CORE_W / 2);
    let (x0, x1) = (600, 600 + CORE_W);
    if ((y0..y1).contains(&j) && i < x1) || ((x0..x1).contains(&i) && j >= y0) { EPS_SI } else { EPS_AIR }
}

struct Interop {
    res: CUgraphicsResource,
    pbo: glow::Buffer,
    _lib: libloading::Library,
}

enum ImagePath {
    /// CUDA writes into the mapped GL pixel buffer; GL copies it into the texture.
    Interop(Interop),
    /// CUDA writes into a device buffer; the host copies it into the texture.
    HostCopy { buf: CudaSlice<u32>, host: Vec<u32> },
}

#[derive(PartialEq, Clone, Copy)]
enum Tool {
    PaintSi,
    Erase,
    Source,
}

/// One line of a `--script` file. The pointer commands become egui pointer
/// events, so they go through the same widgets and hit tests as a mouse.
///   wait N                 N frames
///   ui NAME                click a control: run, step, paint, erase, source, clear
///   uiat NAME FX           click a control at fraction FX of its width (for the gain slider: gain)
///   click I J              click grid cell (I, J)
///   drag I0 J0 I1 J1 N     press at (I0, J0), move over N frames, release at (I1, J1)
///   screenshot FILE        save a screenshot (PPM)
///   quit
enum Cmd {
    Wait(usize),
    Ui(String),
    UiAt(String, f32),
    Click(usize, usize),
    Drag((usize, usize), (usize, usize), usize),
    Screenshot(PathBuf),
    Quit,
}

fn parse_script(text: &str) -> VecDeque<Cmd> {
    let num = |s: Option<&str>| -> usize { s.expect("missing number").parse().expect("bad number") };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut w = l.split_whitespace();
            match w.next().unwrap() {
                "wait" => Cmd::Wait(num(w.next())),
                "ui" => Cmd::Ui(w.next().expect("ui NAME").to_string()),
                "uiat" => Cmd::UiAt(w.next().expect("uiat NAME FX").to_string(), w.next().expect("FX").parse().expect("bad FX")),
                "click" => Cmd::Click(num(w.next()), num(w.next())),
                "drag" => Cmd::Drag((num(w.next()), num(w.next())), (num(w.next()), num(w.next())), num(w.next())),
                "screenshot" => Cmd::Screenshot(PathBuf::from(w.next().expect("screenshot FILE"))),
                "quit" => Cmd::Quit,
                other => panic!("unknown script command {other}"),
            }
        })
        .collect()
}

struct App {
    gpu: GpuYee,
    tex: Option<(glow::Texture, egui::TextureId)>,
    path: Option<ImagePath>,
    path_note: String,
    gl_renderer: String,
    force_host_copy: bool,
    running: bool,
    steps_per_frame: usize,
    tool: Tool,
    brush_r: usize,
    gain: f64,
    source: (usize, usize),
    source_step: usize,
    step: usize,
    rate: (Instant, usize, f64),
    /// --capture: (file, step at which to save it and exit).
    capture: Option<(PathBuf, usize)>,
    /// Requested screenshot: (file, exit after saving).
    shot: Option<(PathBuf, bool)>,
    stroke_cells: usize,
    last_paint: Option<(usize, usize)>,
    script: VecDeque<Cmd>,
    script_wait: usize,
    inject: VecDeque<Vec<egui::Event>>,
    rects: HashMap<&'static str, egui::Rect>,
    image_rect: egui::Rect,
    /// Smoothed wall time of the image path (render kernel + texture update), ms.
    image_ms: f64,
}

fn check(r: CUresult, what: &str) -> Result<(), String> {
    if r == CUresult::CUDA_SUCCESS { Ok(()) } else { Err(format!("{what}: {r:?}")) }
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        force_host_copy: bool,
        capture: Option<(PathBuf, usize)>,
        script: VecDeque<Cmd>,
    ) -> Self {
        let gl = cc.gl.as_ref().expect("glow backend");
        let gl_renderer = unsafe {
            format!("{} / {}", gl.get_parameter_string(glow::VENDOR), gl.get_parameter_string(glow::RENDERER))
        };
        println!("GL: {gl_renderer}");
        let gpu = GpuYee::new(NX, NY, Pml::new(PML), courant(), &initial_eps).expect("CUDA solver");
        println!("CUDA: {}", gpu.device_name());
        Self {
            gpu,
            tex: None,
            path: None,
            path_note: String::new(),
            gl_renderer,
            force_host_copy,
            running: true,
            steps_per_frame: 20,
            tool: Tool::PaintSi,
            brush_r: 8,
            gain: 1.0,
            source: (100, NY / 2),
            source_step: 0,
            step: 0,
            rate: (Instant::now(), 0, 0.0),
            capture,
            shot: None,
            stroke_cells: 0,
            last_paint: None,
            script,
            script_wait: 0,
            inject: VecDeque::new(),
            rects: HashMap::new(),
            image_rect: egui::Rect::NOTHING,
            image_ms: 0.0,
        }
    }

    /// Create the texture and the pixel buffer, then try to register the buffer with CUDA.
    fn init_gl(&mut self, gl: &glow::Context, frame: &mut eframe::Frame) {
        unsafe {
            let tex = gl.create_texture().expect("texture");
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            for (p, v) in [
                (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
                (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, p, v as i32);
            }
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                NX as i32,
                NY as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            self.tex = Some((tex, frame.register_native_glow_texture(tex)));
        }
        let interop = if self.force_host_copy { Err("disabled by --host-copy".to_string()) } else { self.try_interop(gl) };
        self.path = Some(match interop {
            Ok(i) => {
                self.path_note = "CUDA-GL interop: image stays on the GPU".into();
                ImagePath::Interop(i)
            }
            Err(e) => {
                self.path_note = format!("host copy of the RGBA image ({e})");
                ImagePath::HostCopy {
                    buf: self.gpu.stream().alloc_zeros::<u32>(NX * NY).expect("image buffer"),
                    host: vec![0; NX * NY],
                }
            }
        });
        println!("image path: {}", self.path_note);
    }

    fn try_interop(&self, gl: &glow::Context) -> Result<Interop, String> {
        unsafe {
            let pbo = gl.create_buffer()?;
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(pbo));
            gl.buffer_data_size(glow::PIXEL_UNPACK_BUFFER, (NX * NY * 4) as i32, glow::STREAM_DRAW);
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            self.gpu.stream().context().bind_to_thread().map_err(|e| e.to_string())?;
            let lib = libloading::Library::new("libcuda.so.1").map_err(|e| e.to_string())?;
            let register: libloading::Symbol<unsafe extern "C" fn(*mut CUgraphicsResource, u32, u32) -> CUresult> =
                lib.get(b"cuGraphicsGLRegisterBuffer\0").map_err(|e| e.to_string())?;
            let mut res: CUgraphicsResource = std::ptr::null_mut();
            let flags = sys::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD as u32;
            if let Err(e) = check(register(&mut res, pbo.0.get(), flags), "cuGraphicsGLRegisterBuffer") {
                gl.delete_buffer(pbo);
                return Err(e);
            }
            Ok(Interop { res, pbo, _lib: lib })
        }
    }

    /// Render the field into the texture through the active path.
    fn upload(&mut self, gl: &glow::Context) -> Result<(), String> {
        let (tex, _) = self.tex.expect("texture");
        let stream = self.gpu.stream().cu_stream();
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            match self.path.as_mut().expect("image path") {
                ImagePath::Interop(i) => {
                    check(sys::cuGraphicsMapResources(1, &mut i.res, stream), "map")?;
                    let (mut ptr, mut size) = (0u64, 0usize);
                    let mapped = check(sys::cuGraphicsResourceGetMappedPointer_v2(&mut ptr, &mut size, i.res), "pointer");
                    let drawn = mapped.and_then(|_| {
                        if size < NX * NY * 4 {
                            return Err("mapped buffer too small".into());
                        }
                        self.gpu.render_rgba_to_ptr(ptr, self.gain).map_err(|e| e.to_string())
                    });
                    // Unmap also orders the CUDA work before the GL use of the buffer.
                    check(sys::cuGraphicsUnmapResources(1, &mut i.res, stream), "unmap")?;
                    drawn?;
                    gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(i.pbo));
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        NX as i32,
                        NY as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::BufferOffset(0),
                    );
                    gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
                }
                ImagePath::HostCopy { buf, host } => {
                    self.gpu.render_rgba(buf, self.gain).map_err(|e| e.to_string())?;
                    self.gpu.stream().memcpy_dtoh(buf, host.as_mut_slice()).map_err(|e| e.to_string())?;
                    let bytes = std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4);
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D,
                        0,
                        0,
                        0,
                        NX as i32,
                        NY as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(Some(bytes)),
                    );
                }
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        Ok(())
    }

    fn advance(&mut self, steps: usize) {
        let (si, sj) = self.source;
        let (j0, j1) = (sj.saturating_sub(CORE_W / 2).max(1), (sj + CORE_W / 2).min(NY - 1));
        for _ in 0..steps {
            self.gpu.step().expect("step");
            // Continuous wave with a Gaussian turn-on over about 3 periods.
            let t = (self.step - self.source_step + 1) as f64 * courant();
            let ramp = 3.0 * LAMBDA0;
            let env = if t < ramp { (-((t - ramp) / (ramp / 2.0)).powi(2)).exp() } else { 1.0 };
            let amp = env * (2.0 * std::f64::consts::PI * t / LAMBDA0).sin();
            self.gpu.add_source_line(si, j0, j1, amp).expect("source");
            self.step += 1;
        }
    }

    fn cell_at(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(usize, usize)> {
        if !rect.contains(pos) {
            return None;
        }
        let u = (pos.x - rect.left()) / rect.width();
        let v = (pos.y - rect.top()) / rect.height();
        let i = ((u * NX as f32) as usize).min(NX - 1);
        let j = NY - 1 - ((v * NY as f32) as usize).min(NY - 1);
        Some((i, j))
    }

    /// Screen position of the centre of grid cell (i, j); the inverse of `cell_at`.
    fn pos_of(&self, (i, j): (usize, usize)) -> egui::Pos2 {
        let r = self.image_rect;
        egui::pos2(
            r.left() + (i as f32 + 0.5) / NX as f32 * r.width(),
            r.top() + ((NY - 1 - j) as f32 + 0.5) / NY as f32 * r.height(),
        )
    }

    /// Turn the next script command into per-frame batches of egui events.
    fn run_script(&mut self, ctx: &egui::Context) {
        if !self.inject.is_empty() || self.shot.is_some() {
            return;
        }
        if self.script_wait > 0 {
            self.script_wait -= 1;
            return;
        }
        let Some(cmd) = self.script.pop_front() else { return };
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        let click = |pos: egui::Pos2| vec![vec![egui::Event::PointerMoved(pos)], vec![button(pos, true)], vec![button(pos, false)]];
        match cmd {
            Cmd::Wait(n) => self.script_wait = n,
            Cmd::Ui(name) => {
                let r = *self.rects.get(name.as_str()).unwrap_or_else(|| panic!("no control named {name}"));
                println!("script: ui {name}");
                self.inject.extend(click(r.center()));
            }
            Cmd::UiAt(name, fx) => {
                let r = *self.rects.get(name.as_str()).unwrap_or_else(|| panic!("no control named {name}"));
                println!("script: ui {name} at {fx}");
                self.inject.extend(click(egui::pos2(r.left() + fx * r.width(), r.center().y)));
            }
            Cmd::Click(i, j) => {
                println!("script: click cell ({i}, {j})");
                self.inject.extend(click(self.pos_of((i, j))));
            }
            Cmd::Drag(a, b, frames) => {
                println!("script: drag cell {a:?} -> {b:?}");
                let (p0, p1) = (self.pos_of(a), self.pos_of(b));
                self.inject.push_back(vec![egui::Event::PointerMoved(p0)]);
                self.inject.push_back(vec![button(p0, true)]);
                for f in 1..=frames.max(1) {
                    let t = f as f32 / frames.max(1) as f32;
                    self.inject.push_back(vec![egui::Event::PointerMoved(p0 + (p1 - p0) * t)]);
                }
                self.inject.push_back(vec![button(p1, false)]);
            }
            Cmd::Screenshot(path) => self.request_shot(ctx, path, false),
            Cmd::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    fn request_shot(&mut self, ctx: &egui::Context, path: PathBuf, exit: bool) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        self.shot = Some((path, exit));
    }
}

impl eframe::App for App {
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        if let Some(batch) = self.inject.pop_front() {
            raw.events.extend(batch);
        }
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let gl = frame.gl().expect("glow backend").clone();
        if self.tex.is_none() {
            self.init_gl(&gl, frame);
        }
        let stepped = if self.running { self.steps_per_frame } else { 0 };
        self.advance(stepped);
        // Wait for the solver first, so that the timer measures only the image path.
        self.gpu.synchronize().expect("sync");
        let t_img = Instant::now();
        self.upload(&gl).expect("image upload");
        unsafe { gl.finish() };
        let img_ms = t_img.elapsed().as_secs_f64() * 1e3;
        self.image_ms = if self.image_ms == 0.0 { img_ms } else { 0.95 * self.image_ms + 0.05 * img_ms };

        let now = Instant::now();
        self.rate.1 += stepped;
        let dt = now.duration_since(self.rate.0).as_secs_f64();
        if dt > 0.5 {
            self.rate = (now, 0, (self.rate.1 * NX * NY) as f64 / dt / 1e6);
        }

        egui::SidePanel::left("controls").resizable(false).show(ctx, |ui| {
            ui.heading("photon-fdtd");
            ui.horizontal(|ui| {
                let run = ui.button(if self.running { "Pause" } else { "Run" });
                self.rects.insert("run", run.rect);
                if run.clicked() {
                    self.running = !self.running;
                    println!("running = {}", self.running);
                }
                let step = ui.add_enabled(!self.running, egui::Button::new("Step"));
                self.rects.insert("step", step.rect);
                if step.clicked() {
                    self.advance(1);
                    println!("single step -> step {}", self.step);
                }
            });
            ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=200).text("steps / frame"));
            ui.separator();
            ui.label("Mouse tool");
            for (name, tool, label) in [
                ("paint", Tool::PaintSi, "Paint silicon (eps_r 12.11)"),
                ("erase", Tool::Erase, "Erase to air (eps_r 1)"),
                ("source", Tool::Source, "Place source (click)"),
            ] {
                let r = ui.radio_value(&mut self.tool, tool, label);
                self.rects.insert(name, r.rect);
            }
            ui.add(egui::Slider::new(&mut self.brush_r, 1..=40).text("brush radius"));
            ui.separator();
            let gain = ui.add(egui::Slider::new(&mut self.gain, 0.01..=100.0).logarithmic(true).text("gain"));
            self.rects.insert("gain", gain.rect);
            if gain.changed() {
                println!("gain = {:.3}", self.gain);
            }
            let clear = ui.button("Clear fields");
            self.rects.insert("clear", clear.rect);
            if clear.clicked() {
                self.gpu.reset_fields().expect("reset");
                self.source_step = self.step;
                println!("fields cleared at step {}", self.step);
            }
            ui.separator();
            ui.label(format!("grid {NX} x {NY}, CPML {PML} cells"));
            ui.label(format!("step {}  t = {:.0}", self.step, self.step as f64 * courant()));
            ui.label(format!("{:.0} Mcells/s (incl. display)", self.rate.2));
            ui.label(format!("image path {:.3} ms / frame", self.image_ms));
            ui.label(format!("source at ({}, {}), lambda0 {LAMBDA0} cells", self.source.0, self.source.1));
            ui.separator();
            ui.small(format!("GL: {}", self.gl_renderer));
            ui.small(format!("CUDA: {}", self.gpu.device_name()));
            ui.small(&self.path_note);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let scale = (avail.x / NX as f32).min(avail.y / NY as f32);
            let size = egui::vec2(NX as f32 * scale, NY as f32 * scale);
            let (_, id) = self.tex.expect("texture");
            let resp = ui.add(egui::Image::new(egui::load::SizedTexture::new(id, size)).sense(egui::Sense::click_and_drag()));
            self.image_rect = resp.rect;
            let cell = resp.interact_pointer_pos().and_then(|p| self.cell_at(resp.rect, p));
            let brush = matches!(self.tool, Tool::PaintSi | Tool::Erase);
            if brush && resp.is_pointer_button_down_on() && let Some(c) = cell {
                // Paint from the press point, and fill the segment from the last
                // painted cell so that a fast stroke has no gaps.
                let eps = if self.tool == Tool::PaintSi { EPS_SI } else { EPS_AIR };
                let from = self.last_paint.unwrap_or(c);
                let (dx, dy) = (c.0 as f64 - from.0 as f64, c.1 as f64 - from.1 as f64);
                let n = ((dx.hypot(dy) / (self.brush_r as f64 / 2.0).max(1.0)).ceil() as usize).max(1);
                for s in 1..=n {
                    let t = s as f64 / n as f64;
                    let p = ((from.0 as f64 + dx * t).round() as usize, (from.1 as f64 + dy * t).round() as usize);
                    self.gpu.paint_disk(p.0, p.1, self.brush_r, eps).expect("paint");
                    self.stroke_cells += 1;
                }
                if self.last_paint.is_none() {
                    println!("brush eps_r {eps} r {} from cell {c:?}", self.brush_r);
                }
                self.last_paint = Some(c);
            } else if self.last_paint.take().is_some() {
                println!("brush stroke ended after {} disks", self.stroke_cells);
                self.stroke_cells = 0;
            }
            if self.tool == Tool::Source && resp.clicked() && let Some((i, j)) = cell {
                self.source = (i.clamp(PML + 2, NX - PML - 3), j.clamp(PML + CORE_W, NY - PML - CORE_W));
                self.source_step = self.step;
                println!("source placed at ({}, {})", self.source.0, self.source.1);
            }
        });

        if ctx.input(|i| i.key_pressed(egui::Key::F12)) && self.shot.is_none() {
            self.request_shot(ctx, PathBuf::from("out/viewer_f12.ppm"), false);
        }
        if let Some((path, after)) = self.capture.clone()
            && self.step >= after
            && self.shot.is_none()
        {
            self.capture = None;
            self.request_shot(ctx, path, true);
        }
        if self.shot.is_some()
            && let Some(img) = ctx.input(|i| {
                i.raw.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            })
        {
            let (path, exit) = self.shot.take().unwrap();
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            write_ppm(&path, &img);
            println!(
                "screenshot {} at step {} ({:.0} Mcells/s, image path {:.3} ms / frame)",
                path.display(),
                self.step,
                self.rate.2,
                self.image_ms
            );
            if exit {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        self.run_script(ctx);
        ctx.request_repaint();
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(ImagePath::Interop(i)) = self.path.take() {
            unsafe {
                sys::cuGraphicsUnregisterResource(i.res);
                if let Some(gl) = gl {
                    gl.delete_buffer(i.pbo);
                }
            }
        }
        if let (Some((tex, _)), Some(gl)) = (self.tex, gl) {
            unsafe { gl.delete_texture(tex) };
        }
    }
}

fn write_ppm(path: &std::path::Path, img: &egui::ColorImage) {
    let mut out = format!("P6\n{} {}\n255\n", img.size[0], img.size[1]).into_bytes();
    for p in &img.pixels {
        out.extend_from_slice(&[p.r(), p.g(), p.b()]);
    }
    std::fs::write(path, out).expect("write screenshot");
}

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).map(|p| args[p + 1].clone());
    let force_host_copy = args.iter().any(|a| a == "--host-copy");
    let steps: usize = value("--steps").map(|s| s.parse().expect("--steps N")).unwrap_or(3000);
    let capture = value("--capture").map(|p| (PathBuf::from(p), steps));
    let script = value("--script")
        .map(|p| parse_script(&std::fs::read_to_string(&p).expect("read --script file")))
        .unwrap_or_default();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 640.0]).with_title("photon-fdtd viewer"),
        renderer: eframe::Renderer::Glow,
        vsync: true,
        ..Default::default()
    };
    eframe::run_native(
        "photon-fdtd viewer",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, force_host_copy, capture, script)))),
    )
}
