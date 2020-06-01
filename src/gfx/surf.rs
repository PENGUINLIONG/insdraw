use winit::window::Window;
use ash::vk;
use ash::extensions as ashex;
use log::info;
use super::ctxt::Context;
use super::error::{Error, Result};

pub struct Surface<'a> {
    ctxt: &'a Context,
    pub surf: vk::SurfaceKHR,
    entry_khr_surface: ashex::khr::Surface,
}
impl<'a> Surface<'a> {
    pub fn new(ctxt: &'a Context, win: &'a Window) -> Result<Surface<'a>> {
        if cfg!(windows) {
            use winit::platform::windows::WindowExtWindows;

            let hinst = win.hinstance();
            let hwnd = win.hwnd();
            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(hinst)
                .hwnd(hwnd)
                .build();
            let surf = unsafe {
                ashex::khr::Win32Surface::new(&ctxt.entry, &ctxt.inst)
                    .create_win32_surface(&create_info, None)
                    .unwrap()
            };
            info!("created window surface");
            let entry_khr_surface = ashex::khr::Surface::new(&ctxt.entry, &ctxt.inst);
            Ok(Surface { ctxt, surf, entry_khr_surface })
        } else {
            unimplemented!("unsupported platform");
        }
    }
}
impl<'a> Drop for Surface<'a> {
    fn drop(&mut self) {
        unsafe { self.entry_khr_surface.destroy_surface(self.surf, None) };
    }
}
