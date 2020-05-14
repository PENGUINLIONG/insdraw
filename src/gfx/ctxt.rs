use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, Weak};
use std::ops::Range;
use std::marker::PhantomData;
use log::{info, warn, error, debug, trace};
use ash::vk;
use ash::vk_make_version;
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::extensions as vkx;
use lazy_static::lazy_static;
use spirq::{SpirvBinary, EntryPoint as EntryPointManifest, InterfaceLocation,
    Manifest};
use spirq::ty::{DescriptorType};
use winit::window::Window;
use super::error::{Error, Result};
use super::alloc::BuddyAllocator;

const API_VERSION: u32 = vk_make_version!(1, 1, 0);

lazy_static! {
    static ref WANTED_API_EXTS: HashSet<&'static CStr> = {
        [
            vkx::khr::Surface::name(),
            vkx::khr::Win32Surface::name(),
        ].as_ref().into_iter().map(|x| *x).collect()
    };
    
    static ref WANTED_DEV_EXTS: HashSet<&'static CStr> = {
        [
            vkx::khr::Swapchain::name(),
        ].as_ref().into_iter().map(|x| *x).collect()
    };
}

fn filter_exts(
    ext_props: &[vk::ExtensionProperties],
    wanted_exts: &HashSet<&'static CStr>
) -> HashSet<&'static CStr> {
    ext_props.iter()
        .filter_map(|ext_prop| {
            if ext_prop.spec_version <= API_VERSION {
                let name = unsafe {
                    CStr::from_ptr(ext_prop.extension_name.as_ptr())
                };
                wanted_exts.get(name).map(|x| *x)
            } else { None }
        })
        .collect::<HashSet<_>>()
}

macro_rules! try_ext {
    ($a: expr, $b: expr, $enabled_exts: expr, $ext_ty: ty) => {
        if $enabled_exts.contains(<$ext_ty>::name()) {
            Ok(<$ext_ty>::new($a, $b))
        } else { Err(Error::MissingExtension(<$ext_ty>::name())) }
    }
}

macro_rules! def_exts {
    ($ty:ident ($a_name:ident: $a_ty:ty, $b_name:ident: $b_ty:ty) => {$($ext_name:ident: $ext_ty:ty,)*}) => {
        pub(crate) struct $ty {
            $( $ext_name: Option<$ext_ty>, )*
        }
        impl $ty {
            fn new($a_name: &$a_ty, $b_name: &$b_ty, enabled_exts: &HashSet<&CStr>) -> $ty {
                let mut rv = unsafe { MaybeUninit::<Self>::uninit().assume_init() };
                $(
                    if enabled_exts.contains(<$ext_ty>::name()) {
                        rv.$ext_name = Some(<$ext_ty>::new($a_name, $b_name))
                    };
                )*
                rv
            }
            $(
                fn $ext_name(&self) -> Result<&$ext_ty> {
                    self.$ext_name.as_ref()
                        .ok_or(Error::MissingExtension(<$ext_ty>::name()))
                }
            )*
        }
    }
}

def_exts! {
    ApiExtensions(entry: ash::Entry, inst: ash::Instance) => {
        khr_surface: vkx::khr::Surface,
        khr_win32_surface: vkx::khr::Win32Surface,
    }
}

def_exts! {
    DeviceExtensions(inst: ash::Instance, dev: ash::Device) => {
        khr_swapchain: vkx::khr::Swapchain,
    }
}

macro_rules! impl_ptr_wrapper {
    ($ptr_ty: ident -> $inner_ty: ident) => {
        impl std::ops::Deref for $ptr_ty {
            type Target = $inner_ty;
            fn deref(&self) -> &Self::Target { &*self.0 }
        }
        impl std::clone::Clone for $ptr_ty {
            fn clone(&self) -> $ptr_ty { $ptr_ty(self.0.clone()) }
        }
    }
}


fn fmt2aspect(fmt: vk::Format) -> Result<vk::ImageAspectFlags> {
    // TODO: Check this format code limit.
    if fmt.as_raw() >= 1000156000 {
        return Err(Error::UnsupportedPlatform);
    }
    let aspect = match fmt {
        vk::Format::UNDEFINED => Default::default(),
        vk::Format::D16_UNORM |
            vk::Format::X8_D24_UNORM_PACK32 |
            vk::Format::D32_SFLOAT => { vk::ImageAspectFlags::DEPTH },
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT |
            vk::Format::D24_UNORM_S8_UINT |
            vk::Format::D32_SFLOAT_S8_UINT =>
        { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL },
        _ => vk::ImageAspectFlags::COLOR,
    };
    Ok(aspect)
}


pub struct ContextInner {
    // Don't change the order. Things should be dropped from top to bottom.
    inst: ash::Instance,
    entry: ash::Entry,
    cap_detail: ContextCapabilityDetail,
    surf_detail: Option<ContextSurfaceDetail>,
    physdevs: Vec<vk::PhysicalDevice>,
}
impl_ptr_wrapper!(Context -> ContextInner);
pub struct Context(Arc<ContextInner>);
impl Context {
    fn filter_api_exts(entry: &ash::Entry) -> Result<HashSet<&'static CStr>> {
        let ext_props = entry.enumerate_instance_extension_properties()?;
        let api_exts = filter_exts(&ext_props, &WANTED_API_EXTS);
        trace!("wanted and supported instance extensions: {:?}", api_exts);
        Ok(api_exts)
    }
    fn create_inst(
        entry: &ash::Entry,
        app_name: &str,
        api_exts: &HashSet<&'static CStr>,
    ) -> Result<ash::Instance> {
        let app_name = CString::new(app_name).unwrap();
        let engine_name = CString::new("insdraw").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .api_version(API_VERSION)
            .application_name(&app_name)
            .application_version(vk_make_version!(0, 0, 1))
            .engine_name(&engine_name)
            .engine_version(vk_make_version!(0, 0, 1))
            .build();
        let api_exts_raw = api_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let inst_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&api_exts_raw)
            .build();
        let inst = unsafe { entry.create_instance(&inst_create_info, None)? };
        info!("created vulkan instance");
        Ok(inst)
    }
    pub fn new(app_name: &str, wnd: Option<&Window>) -> Result<Context> {
        let entry = ash::Entry::new()?;
        let api_exts = Self::filter_api_exts(&entry)?;
        let inst = Self::create_inst(&entry, app_name, &api_exts)?;
        info!("created context for application '{}'", app_name);

        let cap_detail = ContextCapabilityDetail::new(&entry, &inst, &api_exts);
        let surf_detail = if let Some(wnd) = wnd {
            let surf_detail = ContextSurfaceDetail::new(&cap_detail, wnd)?;
            Some(surf_detail)
        } else { None };
        let physdevs = unsafe { inst.enumerate_physical_devices()? };
        info!("discovered {} physical devices", physdevs.len());

        let inner = ContextInner {
            entry, inst, cap_detail, surf_detail, physdevs
        };
        Ok(Context(Arc::new(inner)))
    }

    pub fn physdevs(&self) -> impl Iterator<Item=Result<PhysicalDevice>> + '_ {
        (0..self.physdevs.len()).into_iter()
            .map(move |i| {
                match PhysicalDevice::new(self, i) {
                    Ok(physdev) => {
                        info!("device #{}: {:?}", i, physdev);
                        Ok(physdev)
                    },
                    Err(e) => {
                        info!("device #{} not accessible", i);
                        Err(e)
                    },
                }
            })
    }
}
impl Drop for ContextInner {
    fn drop(&mut self) {
        if let Some(surf_detail) = self.surf_detail.as_ref() {
            surf_detail.wipe(&self.cap_detail);
        }
        unsafe { self.inst.destroy_instance(None) };
    }
}

struct ContextCapabilityDetail {
    api_exts: ApiExtensions,
}
impl ContextCapabilityDetail {
    pub fn new(
        entry: &ash::Entry,
        inst: &ash::Instance,
        api_exts: &HashSet<&'static CStr>,
    ) -> ContextCapabilityDetail {
        let api_exts = ApiExtensions::new(&entry, &inst, &api_exts);
        ContextCapabilityDetail { api_exts }
    }
}

struct ContextSurfaceDetail {
    surf: vk::SurfaceKHR,
    width: u32,
    height: u32,
}
impl ContextSurfaceDetail {
    fn create_surf(
        ctxt_cap_detail: &ContextCapabilityDetail,
        wnd: &Window,
    ) -> Result<vk::SurfaceKHR> {
        if cfg!(windows) {
            use winit::platform::windows::WindowExtWindows;
            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(wnd.hinstance())
                .hwnd(wnd.hwnd())
                .build();
            let surf = unsafe {
                ctxt_cap_detail.api_exts.khr_win32_surface()?
                    .create_win32_surface(&create_info, None)?
            };
            Ok(surf)
        } else { Err(Error::UnsupportedPlatform) }
    }
    pub fn new(
        ctxt_cap_detail: &ContextCapabilityDetail,
        wnd: &Window,
    ) -> Result<ContextSurfaceDetail> {
        let surf = Self::create_surf(&*ctxt_cap_detail, wnd)?;
        let (width, height): (u32, u32) = wnd.inner_size()
            .to_physical(wnd.hidpi_factor())
            .into();
        let surf_detail = ContextSurfaceDetail { surf, width, height };
        Ok(surf_detail)
    }
    pub fn wipe(&self, ctxt_cap_detail: &ContextCapabilityDetail) {
        unsafe {
            ctxt_cap_detail.api_exts.khr_surface().unwrap()
                .destroy_surface(self.surf, None);
        }
    }
}

pub struct QueueFamily {
    qfam_idx: u32,
    nqueue: u32,
    qflags: vk::QueueFlags,
    present: bool,
}
impl QueueFamily {
    pub fn new(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
        qfam_idx: u32,
        qfam_prop: &vk::QueueFamilyProperties,
    ) -> QueueFamily {
        let nqueue = qfam_prop.queue_count;
        let qflags = qfam_prop.queue_flags;
        let present = ctxt.cap_detail.api_exts.khr_surface.as_ref()
            .and_then(|khr_surface| {
                let surf_detail = ctxt.surf_detail.as_ref()?;
                let support = unsafe {
                    khr_surface.get_physical_device_surface_support(physdev,
                        qfam_idx, surf_detail.surf)
                };
                Some(support)
            }).unwrap_or(false);
        QueueFamily { qfam_idx, nqueue, qflags, present }
    }
}

pub struct PhysicalDeviceInner {
    ctxt: Context,
    physdev: vk::PhysicalDevice,
    pub prop: vk::PhysicalDeviceProperties,
    pub mem_prop: vk::PhysicalDeviceMemoryProperties,
    pub qfams: Vec<QueueFamily>,
    pub dev_exts: HashSet<&'static CStr>,
    pub feats: vk::PhysicalDeviceFeatures,
    surf_detail: Option<PhysicalDeviceSurfaceDetail>,
}
impl_ptr_wrapper!(PhysicalDevice -> PhysicalDeviceInner);
pub struct PhysicalDevice(Arc<PhysicalDeviceInner>);
impl PhysicalDevice {
    fn filter_dev_exts(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
    ) -> Result<HashSet<&'static CStr>> {
        let ext_props = unsafe {
            ctxt.inst.enumerate_device_extension_properties(physdev)?
        };
        let dev_exts = filter_exts(&ext_props, &WANTED_DEV_EXTS);
        trace!("wanted and supported device extensions: {:?}", dev_exts);
        Ok(dev_exts)
    }
    fn filter_feats(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
    ) -> Result<vk::PhysicalDeviceFeatures> {
        let feats = vk::PhysicalDeviceFeatures::default();
        Ok(feats)
    }
    pub fn new(ctxt: &Context, iphysdev: usize) -> Result<PhysicalDevice> {
        let physdev = *ctxt.physdevs.get(iphysdev)
            .ok_or(Error::InvalidOperation)?;

        let dev_exts = Self::filter_dev_exts(ctxt, physdev)?;
        let feats = Self::filter_feats(ctxt, physdev)?;
    
        let prop = unsafe { ctxt.inst.get_physical_device_properties(physdev) };
        let mem_prop = unsafe {
            ctxt.inst.get_physical_device_memory_properties(physdev)
        };
        let qfam_props = unsafe {
            ctxt.inst.get_physical_device_queue_family_properties(physdev)
        };
        let qfams = qfam_props.into_iter()
            .enumerate()
            .map(|(i, prop)| QueueFamily::new(ctxt, physdev, i as u32, &prop))
            .collect();
        let surf_detail = if ctxt.surf_detail.is_some() {
            let surf_detail = PhysicalDeviceSurfaceDetail::new(ctxt, physdev)?;
            Some(surf_detail)
        } else { None };
        let ctxt = ctxt.clone();
        let inner = PhysicalDeviceInner {
            ctxt, physdev, prop, mem_prop, qfams, dev_exts, feats, surf_detail
        };
        Ok(PhysicalDevice(Arc::new(inner)))
    }
}
impl std::fmt::Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let dev_name = unsafe {
            CStr::from_ptr(self.prop.device_name.as_ptr()).to_string_lossy()
        };
        let dev_ty = self.prop.device_type;
        write!(f, "'{}' ({:?})", dev_name, dev_ty)
    }
}

struct PhysicalDeviceSurfaceDetail {
    surf_cap: vk::SurfaceCapabilitiesKHR,
    surf_fmt: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
}
impl PhysicalDeviceSurfaceDetail {
    fn sel_surf_fmt(
        khr_surface: &vkx::khr::Surface,
        physdev: vk::PhysicalDevice,
        surf: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceFormatKHR> {
        let surf_fmts = unsafe {
            khr_surface.get_physical_device_surface_formats(physdev, surf)?
        };
        let fallback = surf_fmts[0].clone();
        let surf_fmt = surf_fmts.into_iter()
            .find(|surf_fmt| surf_fmt.format == vk::Format::B8G8R8A8_UNORM)
            .unwrap_or(fallback);
        Ok(surf_fmt)
    }
    fn sel_present_mode(
        khr_surface: &vkx::khr::Surface,
        physdev: vk::PhysicalDevice,
        surf: vk::SurfaceKHR,
    ) -> Result<vk::PresentModeKHR> {
        let present_modes = unsafe {
            khr_surface.get_physical_device_surface_present_modes(physdev,
                surf)?
        };
        let present_mode = present_modes.into_iter()
            .find(|&present_mode| present_mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        Ok(present_mode)
    }
    pub fn new(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
    ) -> Result<PhysicalDeviceSurfaceDetail> {
        let khr_surface = ctxt.cap_detail.api_exts.khr_surface.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let surf = ctxt.surf_detail.as_ref()
            .ok_or(Error::InvalidOperation)?
            .surf;
        let surf_cap = unsafe {
            khr_surface.get_physical_device_surface_capabilities(physdev, surf)?
        };
        let surf_fmt = Self::sel_surf_fmt(khr_surface, physdev, surf)?;
        let present_mode = Self::sel_present_mode(khr_surface, physdev, surf)?;
        let physdev_surf_detail = PhysicalDeviceSurfaceDetail {
            surf_cap, surf_fmt, present_mode };
        Ok(physdev_surf_detail)
    }
}


struct DeviceQueueDetail {
    qloc: QueueLocation,
    queue: vk::Queue,
}
pub struct DeviceInner {
    physdev: PhysicalDevice,
    dev: ash::Device,
    cap_detail: DeviceCapabilityDetail,
    malloc_detail: DeviceMemoryAllocationDetail,
    present_detail: Option<DevicePresentationDetail>,
    // queue interface -> (queue family index, queue)
    qmap: HashMap<QueueInterface, DeviceQueueDetail>,
}
impl_ptr_wrapper!(Device -> DeviceInner);
pub struct Device(Arc<DeviceInner>);
impl Device {
    fn create_dev(
        physdev: &PhysicalDevice,
        dev_exts: &HashSet<&'static CStr>,
        feats: &vk::PhysicalDeviceFeatures,
        qalloc: &QueueAllocation,
    ) -> Result<ash::Device> {
        let dqcis = qalloc.qfam_alloced.iter()
            .map(|(&qfam_idx, &nqueue)| {
                let priors = (0..nqueue).into_iter()
                    .map(|_| 0.5)
                    .collect::<Vec<_>>();
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(qfam_idx)
                    .queue_priorities(&priors)
                    .build()
            })
            .collect::<Vec<_>>();
        let dev_exts_raw = dev_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let create_info = vk::DeviceCreateInfo::builder()
            .enabled_features(&feats)
            .queue_create_infos(&dqcis)
            .enabled_extension_names(&dev_exts_raw)
            .build();
        let dev = unsafe {
            physdev.ctxt.inst.create_device(physdev.physdev, &create_info,
                None)?
        };
        info!("created device on {:?}", physdev);
        Ok(dev)
    }
    fn collect_qmap(
        dev: &ash::Device,
        qalloc: &QueueAllocation,
    ) -> HashMap<QueueInterface, DeviceQueueDetail> {
        qalloc.qloc_assign.iter()
            .filter_map(|(qi, qloc)| {
                if let Some(qloc) = qloc {
                    info!("set up queue for interface '{:?}'", qi);
                    let queue = unsafe {
                        dev.get_device_queue(qloc.qfam_idx, qloc.queue_idx)
                    };
                    let queue_detail = DeviceQueueDetail {
                        qloc: qloc.clone(),
                        queue,
                    };
                    Some((*qi, queue_detail))
                } else {
                    warn!("interface '{:?}' is not available", qi);
                    None
                }
            })
            .collect::<HashMap<_, _>>()
    }
    pub fn new(physdev: &PhysicalDevice) -> Result<Device> {
        let qalloc = QueueAllocation::new(physdev);
        let dev = Self::create_dev(physdev, &physdev.dev_exts, &physdev.feats,
            &qalloc)?;

        let qmap = Self::collect_qmap(&dev, &qalloc);

        // Pre-creation details. For those details which doesn't depend on the
        // device object as params.
        let cap_detail = {
            DeviceCapabilityDetail::new(physdev, &dev, &physdev.dev_exts,
                &physdev.feats)
        };
        let malloc_detail = {
            let trans_queue = qmap.get(&QueueInterface::Transfer)
                .map(|x| x.queue);
            DeviceMemoryAllocationDetail::new(physdev, &dev, trans_queue)?
        };
        let present_detail = {
            if qmap.contains_key(&QueueInterface::Present) {
                Some(
                    DevicePresentationDetail::new(physdev, &cap_detail, &qmap)?
                )
            } else {
                None
            }
        };
        
        // Device object creation.
        let inner = DeviceInner {
            physdev: physdev.clone(),
            dev,
            cap_detail,
            malloc_detail,
            present_detail,
            qmap,
        };
        let inner = Arc::new(inner);

        Ok(Device(inner))
    }

    /// Allocate memory on the device. Small memory chunks will be paged while
    /// large memory chunks are provided with dedicated allocation.
    fn alloc_mem(
        dev: Arc<DeviceInner>,
        mem_req: &vk::MemoryRequirements,
        mem_usage: MemoryUsage,
    ) -> Result<MemorySlice> {
        let size = mem_req.size as usize;
        let align = mem_req.alignment as usize;
        let mem_ty = &dev.malloc_detail.mem_ty_sel[&mem_usage]
            .match_mem_ty(mem_req.memory_type_bits)
            .ok_or(Error::UnsupportedPlatform)?;
        if size > PagedMemoryAllocator::PAGE_SIZE {
            // Dedicated allocation.
            let mem = Arc::new(Memory::new(dev.clone(), mem_ty, size, None)?);
            Ok(MemorySlice::new(&mem, 0, size))
        } else {
            // Paged allocation.
            let weak_dev = Arc::downgrade(&dev);
            dev.malloc_detail.pallocs
                .lock()
                .unwrap()
                .entry(mem_ty.mem_ty_idx)
                .or_insert_with(|| PagedMemoryAllocator::new(mem_ty))
                .alloc(weak_dev, size, align)
        }
    }
    fn alloc_buf_mem(
        dev: Arc<DeviceInner>,
        buf: vk::Buffer,
        mem_usage: MemoryUsage,
    ) -> Result<MemorySlice> {
        let mem_req = unsafe { dev.dev.get_buffer_memory_requirements(buf) };
        Self::alloc_mem(dev.clone(), &mem_req, mem_usage)
            .and_then(|mem| unsafe {
                let offset = mem.offset as u64;
                let dev_mem = mem.mem.dev_mem;
                dev.dev.bind_buffer_memory(buf, dev_mem, offset)?;
                Ok(mem)
            })
    }
    fn alloc_img_mem(
        dev: Arc<DeviceInner>,
        img: vk::Image,
        mem_usage: MemoryUsage,
    ) -> Result<MemorySlice> {
        let mem_req = unsafe { dev.dev.get_image_memory_requirements(img) };
        Self::alloc_mem(dev.clone(), &mem_req, mem_usage)
            .and_then(|mem| unsafe {
                let offset = mem.offset as u64;
                let dev_mem = mem.mem.dev_mem;
                dev.dev.bind_image_memory(img, dev_mem, offset)?;
                Ok(mem)
            })
    }

    pub fn acquire_swapchain_img(
        &self,
        timeout: u64,
    ) -> Result<SwapchainImage> {
        let (present_detail, khr_swapchain) = {
            let dev = &*self.0;
            (
                dev.present_detail.as_ref()
                    .ok_or(Error::InvalidOperation)?,
                dev.cap_detail.dev_exts.khr_swapchain.as_ref()
                    .ok_or(Error::InvalidOperation)?,
            )
        };

        let (img_idx, _is_suboptimal) = unsafe {
            khr_swapchain.acquire_next_image(present_detail.swapchain, timeout,
                vk::Semaphore::null(), vk::Fence::null())?
        };
        let img = present_detail.imgs[img_idx as usize];
        let img_cfg = present_detail.img_cfg.clone();
        let img = Image::new_implicit(self, img_cfg, img)?;
        Ok(SwapchainImage { img, img_idx })
    }
}
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
enum QueueInterface {
    Graphics,
    Compute,
    Transfer,
    Present,
}
#[derive(Clone)]
struct QueueLocation {
    /// Index of the queue family of the queue.
    qfam_idx: u32,
    /// Index of the queue within the queue family.
    queue_idx: u32,
}
struct QueueAllocation {
    /// Location assigned to queue interfaces.
    qloc_assign: HashMap<QueueInterface, Option<QueueLocation>>,
    /// Mapping from queue family index to number of queues allocated in the
    /// family.
    qfam_alloced: HashMap<u32, u32>,
}
impl QueueAllocation {
    fn collect_qfam_idxs_by_cap(
        physdev: &PhysicalDevice,
    ) -> [(QueueInterface, HashSet<u32>); 4] {
        let qfams = &physdev.qfams;
        let nqfam = qfams.len();
        let graph_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].qflags.intersects(vk::QueueFlags::GRAPHICS) {
                    Some(i as u32)
                } else { None }
            })
            .collect::<HashSet<_>>();
        let comp_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].qflags.intersects(vk::QueueFlags::COMPUTE) {
                    Some(i as u32)
                } else { None }
            })
            .collect::<HashSet<_>>();
        let trans_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].qflags.intersects(vk::QueueFlags::TRANSFER) {
                    Some(i as u32)
                } else { None }
            })
            .collect::<HashSet<_>>();
        let present_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].present { Some(i as u32) } else { None }
            })
            .collect::<HashSet<_>>();
        [
            (QueueInterface::Graphics, graph_qfam_idxs),
            (QueueInterface::Compute, comp_qfam_idxs),
            (QueueInterface::Transfer, trans_qfam_idxs),
            (QueueInterface::Present, present_qfam_idxs),
        ]
    }
    pub fn new(physdev: &PhysicalDevice) -> QueueAllocation {
        // Mapping from queue family to the number of queues has been allocated
        // in that family.
        let mut qloc_assign: HashMap<QueueInterface, Option<QueueLocation>> = {
            HashMap::new()
        };
        let mut qfam_alloced: HashMap<u32, u32> = HashMap::new();
        let mut cap_qfams = Self::collect_qfam_idxs_by_cap(physdev);
        // Satify scarce resource requirements first.
        cap_qfams.sort_by_key(|x| x.1.len());
        for cap_qfam in cap_qfams.iter() {
            let qi = cap_qfam.0;
            // Find a queue family that hasn't been filled up.
            if let Some(qloc) = cap_qfam.1.iter()
                .find_map(|&qfam_idx| {
                    let nqueue = physdev.qfams[qfam_idx as usize].nqueue;
                    let queue_idx = *qfam_alloced.get(&qfam_idx).unwrap_or(&0);
                    if queue_idx < nqueue {
                        Some(QueueLocation { qfam_idx, queue_idx })
                    } else { None }
                })
            {
                // Allocate available queue to the capability.
                *qfam_alloced.entry(qloc.qfam_idx).or_default() += 1;
                qloc_assign.insert(qi, Some(qloc));
            } else {
                qloc_assign.insert(qi, None);
            }
        }
        QueueAllocation { qloc_assign, qfam_alloced }
    }
}

struct DeviceCapabilityDetail {
    /// Enabled wanted device extensions.
    dev_exts: DeviceExtensions,
    /// Enabled wanted device features.
    feats: vk::PhysicalDeviceFeatures,
}
impl DeviceCapabilityDetail {
    pub fn new(
        physdev: &PhysicalDevice,
        dev: &ash::Device,
        dev_exts: &HashSet<&'static CStr>,
        feats: &vk::PhysicalDeviceFeatures,
    ) -> DeviceCapabilityDetail {
        let dev_exts = DeviceExtensions::new(&physdev.ctxt.inst, dev, dev_exts);
        let feats = feats.clone();
        DeviceCapabilityDetail { dev_exts, feats }
    }
}

// ------ Memory allocation ------

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct MemoryProperty(vk::MemoryPropertyFlags);
impl MemoryProperty {
    #[inline]
    fn device_local_bit(&self) -> u32 {
        self.0.as_raw() & vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
    }
    #[inline]
    fn host_cached_bit(&self) -> u32 {
        self.0.as_raw() & vk::MemoryPropertyFlags::HOST_CACHED.as_raw()
    }
    #[inline]
    fn host_visible_bit(&self) -> u32 {
        self.0.as_raw() & vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
    }
    #[inline]
    fn host_coherent_bit(&self) -> u32 {
        self.0.as_raw() & vk::MemoryPropertyFlags::HOST_COHERENT.as_raw()
    }

    pub fn push_score(&self) -> u32 {
        self.host_visible_bit() << 3 |
            self.host_coherent_bit() << 2 |
            self.device_local_bit() << 1 |
            self.host_cached_bit()
    }
    pub fn pull_score(&self) -> u32 {
        self.host_visible_bit() << 3 |
            self.host_cached_bit() << 2 |
            self.host_coherent_bit() << 1 |
            self.device_local_bit()
    }
    pub fn dev_score(&self) -> u32 {
        self.device_local_bit() << 3 |
            self.host_visible_bit() << 2 |
            self.host_coherent_bit() << 1 |
            self.host_cached_bit()
    }
}
#[derive(Debug, Clone)]
struct MemoryType {
    mem_ty_idx: u32,
    mem_prop: MemoryProperty,
}


struct MemoryInner {
    dev: Arc<DeviceInner>,
    dev_mem: vk::DeviceMemory,
    size: usize,
    mem_prop: MemoryProperty,
    malloc: Option<Mutex<BuddyAllocator>>,
}
impl_ptr_wrapper!(Memory -> MemoryInner);
struct Memory(Arc<MemoryInner>);
impl Memory {
    fn alloc_dev_mem(
        dev: &DeviceInner,
        mem_ty: &MemoryType,
        size: usize,
    ) -> Result<vk::DeviceMemory> {
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size as u64)
            .memory_type_index(mem_ty.mem_ty_idx)
            .build();
        info!("allocated device memory of {} bytes", size);
        let dev_mem = unsafe { dev.dev.allocate_memory(&alloc_info, None)? };
        Ok(dev_mem)
    }
    fn new(
        dev: Arc<DeviceInner>,
        mem_ty: &MemoryType,
        size: usize,
        malloc: Option<BuddyAllocator>,
    ) -> Result<Memory> {
        let dev_mem = Self::alloc_dev_mem(&*dev, mem_ty, size)?;
        let mem_prop = mem_ty.mem_prop.clone();
        let malloc = malloc.map(|x| Mutex::new(x));
        let inner = MemoryInner { dev, dev_mem, size, mem_prop, malloc };
        Ok(Memory(Arc::new(inner)))
    }
}
impl Drop for MemoryInner {
    fn drop(&mut self) {
        unsafe { self.dev.dev.free_memory(self.dev_mem, None) };
    }
}

/// A section of memory in a large allocation.
#[derive(Clone)]
pub struct MemorySlice {
    mem: Memory,
    /// Offset of the suballocation in the owner memory page. `page_offset` is
    /// always less than any current offset of the slice; and `offset + size` is
    /// always within the bound of the allocated range. E.g.
    ///
    /// ```
    /// page_offset->|<- allocated size ->|
    ///           offset->|<- size ->|
    /// ```
    page_offset: usize,
    offset: usize,
    size: usize,
}
impl MemorySlice {
    /// Create a new memory slice and refer to the `offset` if `mem` is paged.
    /// This should be used internally only.
    fn new(mem: &Memory, offset: usize, size: usize) -> MemorySlice {
        if let Some(malloc) = mem.malloc.as_ref() {
            // Increase the reference count at the referred intra-page location
            // immediately.
            malloc.lock().unwrap().refer(offset);
        }
        let mem = mem.clone();
        MemorySlice { mem, page_offset: offset, offset, size }
    }
    pub fn slice(&self, offset: usize, size: usize) -> Result<MemorySlice> {
        if offset + size <= self.size {
            let mem = MemorySlice {
                mem: self.mem.clone(),
                page_offset: self.page_offset,
                offset: self.offset + offset,
                size: size,
            };
            Ok(mem)
        } else {
            Err(Error::InvalidOperation)
        }
    }
    pub fn write(&self, src: &[u8]) -> Result<()> {
        let dev_mem = self.mem.dev_mem;
        let dev = &self.mem.dev.dev;
        unsafe {
            let offset = self.offset as u64;
            let dst = dev.map_memory(
                dev_mem,
                offset,
                self.size as u64,
                vk::MemoryMapFlags::empty())? as *mut u8;
            std::intrinsics::copy(src.as_ptr(), dst, self.size);
            dev.unmap_memory(dev_mem);
        }
        Ok(())
    }
    pub fn read(&self, dst: &mut [u8]) -> Result<()> {
        let dev_mem = self.mem.dev_mem;
        let dev = &self.mem.dev.dev;
        unsafe {
            let offset = self.offset as u64;
            let src = dev.map_memory(
                dev_mem,
                offset,
                self.size as u64,
                vk::MemoryMapFlags::empty())? as *mut u8;
            std::intrinsics::copy(src, dst.as_mut_ptr(), self.size);
            dev.unmap_memory(dev_mem);
        }
        Ok(())
    }
}
impl Drop for MemorySlice {
    fn drop(&mut self) {
        if let Some(malloc) = self.mem.malloc.as_ref() {
            // Decrease the reference count at the location.
            malloc.lock().unwrap().free(self.page_offset);
        }
    }
}

/// Manager of small object allocation.
struct PagedMemoryAllocator {
    pages: Vec<Memory>,
    pos: usize,
    mem_ty: MemoryType,
}
impl PagedMemoryAllocator {
    pub const PAGE_SIZE: usize = 8 * 1024 * 1024;
    pub const HALF_PAGE_SIZE: usize = Self::PAGE_SIZE / 2;
    pub const PAGE_MAX_ORDER: u32 = 15;

    pub fn new(mem_ty: &MemoryType) -> PagedMemoryAllocator {
        PagedMemoryAllocator {
            pages: Vec::new(),
            pos: 0,
            mem_ty: mem_ty.clone(),
        }
    }
    /// Allocate on memory page indexed by `i`.
    fn alloc_on(
        &mut self,
        i: usize,
        size: usize,
        align: usize,
    ) -> Option<MemorySlice> {
        let cur_page = &mut self.pages[i];
        let addr = {
            cur_page.malloc.as_ref().unwrap().lock().unwrap()
                .alloc(size, align)
        };
        addr.map(|offset| MemorySlice::new(&cur_page, offset, size))
    }
    /// Allocate on a newly created page, added at the end of the page list.
    /// This method doesn't take alignment because the suballocation will
    /// always taken at offset=0.
    fn alloc_on_new_page(
        &mut self,
        dev: Weak<DeviceInner>,
        size: usize,
    ) -> Result<MemorySlice> {
        // Out of paged memory. We have to allocate the memory on a new
        // page.
        let dev = dev.upgrade()
            .ok_or(Error::InvalidOperation)?;
        let mem = {
            let malloc = {
                Some(BuddyAllocator::new(Self::PAGE_SIZE, Self::PAGE_MAX_ORDER))
            };
            Memory::new(dev, &self.mem_ty, Self::PAGE_SIZE, malloc)?
        };
        self.pos = self.pages.len();
        self.pages.push(mem);
        info!("created memory page");
        // This should not fail, unless the allocation size is greater than
        // `PAGE_SIZE` which is not allowed.
        self.alloc_on(self.pos, size, 0)
            .ok_or(Error::InvalidOperation)
    }
    /// Allocate on memory page. A new page is allocated if the allocation
    /// cannot be done on existing pages. Size of memory to be allocated MUST be
    /// less than `PAGE_SIZE`.
    pub fn alloc(
        &mut self,
        dev: Weak<DeviceInner>,
        size: usize,
        align: usize,
    ) -> Result<MemorySlice> {
        // Attempt to allocatef on existing page.
        let front = self.pos..self.pages.len();
        let back = 0..self.pos;
        front.into_iter().chain(back.into_iter())
            .find_map(|i| self.alloc_on(i, size, align))
            .ok_or(Error::OutOfMemory)
            .or_else(|_| self.alloc_on_new_page(dev, size))
    }
    /// Drop all unused memory pages.
    pub fn shrink_to_fit(&mut self, dev: &Device) {
        let mut i = 0;
        while i < self.pages.len() {
            let unused = {
                let malloc = self.pages[i].malloc.as_ref().unwrap();
                let malloc = malloc.lock().unwrap();
                malloc.is_unused()
            };
            if unused {
                self.pages.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum MemoryUsage {
    /// Constant data container which is seldom written by the host but
    /// potentially frequently read by the device. I/O with device memory might
    /// incur staging memory allocations.
    ///
    /// NOTE: It's NOT guaranteed the memory will be allocated on a device-local
    /// heap.
    Device,
    /// Pushing data container which is frequently written by the host and
    /// potentially frequently read by the device.
    Push,
    /// Pulling data container which is frequently read by the host and
    /// potentially frequently written by the device.
    Pull,
}

struct MemoryTypeFallbackChain(Vec<MemoryType>);
impl MemoryTypeFallbackChain {
    fn match_mem_ty(&self, mem_ty_bits: u32) -> Option<&MemoryType> {
        self.0.iter()
            .find(|mem_ty| mem_ty_bits & 1 << mem_ty.mem_ty_idx != 0)
    }
}
#[derive(Default)]
struct MemoryStaging {
    trans_queue: vk::Queue,
    push: Mutex<Option<MemoryInner>>,
    pull: Mutex<Option<MemoryInner>>,
}
struct DeviceMemoryAllocationDetail {
    mem_ty_sel: HashMap<MemoryUsage, MemoryTypeFallbackChain>,
    stage: Option<MemoryStaging>,
    /// Allocator for each memory type. The indices of allocator correspond to
    /// memory type indices.
    pallocs: Mutex<HashMap<u32, PagedMemoryAllocator>>,
}
impl DeviceMemoryAllocationDetail {
    fn collect_mem_ty_sel(
        physdev: &PhysicalDevice,
        dev: &ash::Device,
    ) -> Result<HashMap<MemoryUsage, MemoryTypeFallbackChain>> {
        let nmem_ty = physdev.mem_prop.memory_type_count as usize;
        let mut has_host_visible = 0;
        let mut mem_tys = physdev.mem_prop.memory_types[..nmem_ty]
            .into_iter()
            .enumerate()
            .map(|(mem_ty_idx, mem_ty)| {
                let mem_ty_idx = mem_ty_idx as u32;
                let mem_prop = MemoryProperty(mem_ty.property_flags);
                has_host_visible |= mem_prop.host_visible_bit();
                MemoryType { mem_ty_idx, mem_prop }
            })
            .collect::<Vec<_>>();

        mem_tys.sort_by_key(|mem_ty| mem_ty.mem_prop.dev_score());
        let dev_tys = mem_tys.iter()
            .cloned()
            .collect::<Vec<_>>();

        mem_tys.sort_by_key(|mem_ty| mem_ty.mem_prop.push_score());
        let push_tys = mem_tys.iter()
            .take_while(|mem_ty| mem_ty.mem_prop.host_visible_bit() != 0)
            .cloned()
            .collect::<Vec<_>>();

        mem_tys.sort_by_key(|mem_ty| mem_ty.mem_prop.pull_score());
        let pull_tys = mem_tys.iter()
            .take_while(|mem_ty| mem_ty.mem_prop.host_visible_bit() != 0)
            .cloned()
            .collect::<Vec<_>>();

        if has_host_visible == 0 {
            // There is no host visible memory type. We cannot pass data to the
            // device.
            return Err(Error::UnsupportedPlatform);
        }

        let mut sel = HashMap::new();
        sel.insert(MemoryUsage::Device, MemoryTypeFallbackChain(dev_tys));
        sel.insert(MemoryUsage::Push, MemoryTypeFallbackChain(push_tys));
        sel.insert(MemoryUsage::Pull, MemoryTypeFallbackChain(pull_tys));
        Ok(sel)
    }
    pub fn new(
        physdev: &PhysicalDevice,
        dev: &ash::Device,
        trans_queue: Option<vk::Queue>,
    ) -> Result<DeviceMemoryAllocationDetail> {
        let mem_ty_sel = Self::collect_mem_ty_sel(physdev, dev)?;
        let stage = trans_queue
            .map(|q| MemoryStaging { trans_queue: q, ..Default::default() });
        let pallocs = Default::default();
        let malloc_detail = DeviceMemoryAllocationDetail { mem_ty_sel, stage,
            pallocs };
        Ok(malloc_detail)
    }
}

struct DevicePresentationDetail {
    swapchain: vk::SwapchainKHR,
    imgs: Vec<vk::Image>,
    img_cfg: ImageConfig,
}
impl DevicePresentationDetail {
    const PREFERRED_IMG_COUNT: u32 = 3;
    fn create_swapchain(
        physdev: &PhysicalDevice,
        khr_swapchain: &vkx::khr::Swapchain,
        qmap: &HashMap<QueueInterface, DeviceQueueDetail>,
    ) -> Result<(vk::SwapchainKHR, ImageConfig)> {
        let physdev_surf_detail = physdev.surf_detail.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let ctxt_surf_detail = physdev.ctxt.surf_detail.as_ref()
            .ok_or(Error::InvalidOperation)?;

        let nimg = Self::PREFERRED_IMG_COUNT
            .min(physdev_surf_detail.surf_cap.max_image_count);
        let fmt = physdev_surf_detail.surf_fmt.format;
        let color_space = physdev_surf_detail.surf_fmt.color_space;
        let trans = physdev_surf_detail.surf_cap.current_transform;

        let width = ctxt_surf_detail.width;
        let height = ctxt_surf_detail.height;

        let present_qfam_idx = qmap.get(&QueueInterface::Present)
            .map(|x| x.qloc.qfam_idx)
            .unwrap();
        let graph_qfam_idx = qmap.get(&QueueInterface::Graphics)
            .map(|x| x.qloc.qfam_idx)
            .unwrap_or(!0);
        let qfam_idxs = [present_qfam_idx, graph_qfam_idx];
        let (share_mode, nqfam_idx) = if qfam_idxs[1] == !0 {
            warn!("present is enabled while graphics is not");
            (vk::SharingMode::EXCLUSIVE, 1)
        } else {
            (vk::SharingMode::CONCURRENT, 2)
        };
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(ctxt_surf_detail.surf)
            .min_image_count(nimg)
            .image_format(fmt)
            .image_color_space(color_space)
            .image_extent(vk::Extent2D { width, height })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(share_mode)
            .queue_family_indices(&qfam_idxs[..nqfam_idx])
            .present_mode(physdev_surf_detail.present_mode)
            .clipped(true)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .build();

        let swapchain = unsafe {
            khr_swapchain.create_swapchain(&create_info, None)?
        };
        info!("created swapchain");

        let img_cfg = ImageConfig {
            fmt, view_ty: vk::ImageViewType::TYPE_2D, width, height, depth: 1,
            nlayer: 1, nmip: 1, usage: vk::ImageUsageFlags::COLOR_ATTACHMENT 
        };
        Ok((swapchain, img_cfg))
    }
    pub fn new(
        physdev: &PhysicalDevice,
        cap_detail: &DeviceCapabilityDetail,
        qmap: &HashMap<QueueInterface, DeviceQueueDetail>,
    ) -> Result<DevicePresentationDetail> {
        let khr_swapchain = cap_detail.dev_exts.khr_swapchain.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let (swapchain, img_cfg) = Self::create_swapchain(physdev,
            khr_swapchain, &qmap)?;
        let imgs = unsafe { khr_swapchain.get_swapchain_images(swapchain)? };

        let present_detail = DevicePresentationDetail {
            swapchain, imgs, img_cfg
        };
        Ok(present_detail)
    }
    pub fn wipe(&mut self, cap_detail: &DeviceCapabilityDetail) {
        let khr_swapchain = cap_detail.dev_exts.khr_swapchain.as_ref()
            .unwrap();
        unsafe { khr_swapchain.destroy_swapchain(self.swapchain, None) };
    }
}

pub struct SwapchainImage {
    pub img: Image,
    pub img_idx: u32,
}


struct DeviceGraphicsDetail {
    graph_queue: vk::Queue,
}
impl DeviceGraphicsDetail {
    pub fn new(graph_queue: vk::Queue) -> Result<DeviceGraphicsDetail> {
        let graph_detail = DeviceGraphicsDetail { graph_queue };
        Ok(graph_detail)
    }
}


struct DeviceComputeDetail {
    comp_queue: vk::Queue,
}
impl DeviceComputeDetail {
    pub fn new(comp_queue: vk::Queue) -> Result<DeviceComputeDetail> {
        let comp_detail = DeviceComputeDetail { comp_queue };
        Ok(comp_detail)
    }
}

enum MemoryManagement {
    /// Resource has been managed by us (insdraw). We can access the memory
    /// content directly.
    Explicit(MemorySlice),
    /// Resource has been managed by vulkan core or extensions internally, we
    /// don't have direct access into the memory content of the resource.
    /// Swapchain images are in this type.
    Implicit(Arc<DeviceInner>),
}
impl MemoryManagement {
    pub fn dev(&self) -> &Arc<DeviceInner> {
        match &self {
            MemoryManagement::Explicit(mem) => &mem.mem.dev,
            MemoryManagement::Implicit(dev) => dev,
        }
    }
}

#[derive(Clone)]
pub struct BufferConfig {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
}
impl_ptr_wrapper!(Buffer -> BufferInner);
pub struct Buffer(Arc<BufferInner>);
pub struct BufferInner {
    buf: vk::Buffer,
    mem: MemoryManagement,
    cfg: BufferConfig,
}
impl Buffer {
    fn create_buf(
        dev: &DeviceInner,
        buf_cfg: &BufferConfig,
    ) -> Result<vk::Buffer> {
        if buf_cfg.usage.as_raw() == 0 { return Err(Error::InvalidOperation) }
        let create_info = vk::BufferCreateInfo::builder()
            .size(buf_cfg.size as u64)
            .usage(buf_cfg.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let buf = unsafe { dev.dev.create_buffer(&create_info, None)? };
        info!("created buffer");
        Ok(buf)
    }
    pub fn new(
        dev: &Device,
        buf_cfg: BufferConfig,
        mem_usage: MemoryUsage,
    ) -> Result<Buffer> {
        let dev = dev.0.clone();
        let buf = Self::create_buf(&*dev, &buf_cfg)?;
        let mem = Device::alloc_buf_mem(dev, buf, mem_usage)?;
        let mem = MemoryManagement::Explicit(mem);
        let inner = BufferInner { buf, mem, cfg: buf_cfg };
        Ok(Buffer(Arc::new(inner)))
    }
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        match &self.mem {
            MemoryManagement::Explicit(buf) => {
                let dev = &self.mem.dev().dev;
                unsafe { dev.destroy_buffer(self.buf, None) };
                info!("destroyed buffer");
            },
            MemoryManagement::Implicit(buf) => {
                unimplemented!();
            }
        }
    }
}

#[derive(Clone)]
pub struct ImageConfig {
    pub fmt: vk::Format,
    pub view_ty: vk::ImageViewType,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub nlayer: u32,
    pub nmip: u32,
    pub usage: vk::ImageUsageFlags,
}
pub struct ImageInner {
    img: vk::Image,
    img_view: vk::ImageView,
    // Swapchain images have no explicitly allocated memory.
    mem: MemoryManagement,
    cfg: ImageConfig,
    layout: Cell<vk::ImageLayout>,
    aspect: vk::ImageAspectFlags,
}
impl_ptr_wrapper!(Image -> ImageInner);
pub struct Image(Arc<ImageInner>);
impl Image {
    fn create_img(
        dev: &DeviceInner,
        img_cfg: &ImageConfig
    ) -> Result<vk::Image> {
        if img_cfg.usage.as_raw() == 0 { return Err(Error::InvalidOperation) }
        let flags = match img_cfg.view_ty {
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            },
            _ => vk::ImageCreateFlags::empty(),
        };
        let img_ty = match img_cfg.view_ty {
            vk::ImageViewType::TYPE_1D => vk::ImageType::TYPE_1D,
            vk::ImageViewType::TYPE_2D |
                vk::ImageViewType::CUBE |
                vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageType::TYPE_2D,
            vk::ImageViewType::TYPE_3D |
                vk::ImageViewType::TYPE_2D_ARRAY |
                vk::ImageViewType::CUBE_ARRAY => vk::ImageType::TYPE_2D,
            _ => unreachable!(),
        };
        let extent = vk::Extent3D {
            width: img_cfg.width,
            height: img_cfg.height,
            depth: img_cfg.depth,
        };
        let create_info = vk::ImageCreateInfo::builder()
            .flags(flags)
            .image_type(img_ty)
            .format(img_cfg.fmt)
            .extent(extent)
            .mip_levels(img_cfg.nmip)
            .array_layers(img_cfg.nlayer)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(img_cfg.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let img = unsafe { dev.dev.create_image(&create_info, None)? };
        info!("created image");
        Ok(img)
    }
    fn create_img_view(
        dev: &ash::Device,
        img_cfg: &ImageConfig,
        aspect: vk::ImageAspectFlags,
        img: vk::Image,
    ) -> Result<vk::ImageView> {
        let subrsc_rng = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: img_cfg.nmip,
            base_array_layer: 0,
            layer_count: img_cfg.nlayer,
        };
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(img)
            .view_type(img_cfg.view_ty)
            .format(img_cfg.fmt)
            .subresource_range(subrsc_rng)
            .build();
        let img_view = unsafe { dev.create_image_view(&create_info, None)? };
        Ok(img_view)
    }
    pub fn new(
        dev: &Device,
        img_cfg: ImageConfig,
        mem_use: MemoryUsage,
    ) -> Result<Image> {
        let dev = dev.0.clone();
        let img = Self::create_img(&*dev, &img_cfg)?;
        let mem = Device::alloc_img_mem(dev.clone(), img, mem_use)?;
        let mem = MemoryManagement::Explicit(mem);
        let aspect = fmt2aspect(img_cfg.fmt)?;
        let img_view = Self::create_img_view(&dev.dev, &img_cfg, aspect, img)?;
        let layout = Cell::new(vk::ImageLayout::UNDEFINED);
        let inner = ImageInner {
            img, img_view, mem, cfg: img_cfg, layout, aspect,
        };
        Ok(Image(Arc::new(inner)))
    }
    fn new_implicit(
        dev: &Device,
        img_cfg: ImageConfig,
        img: vk::Image,
    ) -> Result<Image> {
        let dev = dev.0.clone();
        let aspect = fmt2aspect(img_cfg.fmt)?;
        let img_view = Self::create_img_view(&dev.dev, &img_cfg, aspect, img)?;
        let layout = Cell::new(vk::ImageLayout::UNDEFINED);
        let mem = MemoryManagement::Implicit(dev);
        let inner = ImageInner {
            img, img_view, mem, cfg: img_cfg, layout, aspect,
        };
        Ok(Image(Arc::new(inner)))
    }
}
impl Drop for ImageInner {
    fn drop(&mut self) {
        match &self.mem {
            MemoryManagement::Explicit(mem) => {
                let dev = &self.mem.dev().dev;
                unsafe {
                    dev.destroy_image(self.img, None);
                    info!("destroyed image");
                    dev.destroy_image_view(self.img_view, None);
                    info!("destroyed image view");
                }
            },
            MemoryManagement::Implicit(dev) => {
                unsafe { dev.dev.destroy_image_view(self.img_view, None) };
                info!("destroyed image view");
            },
        }
    }
}

#[derive(Clone)]
struct SamplerConfig {
    // TODO: (penguinliong) impl
}
pub struct SamplerInner {
    dev: Arc<DeviceInner>,
    sampler: vk::Sampler,
    sampler_cfg: SamplerConfig,
}
impl_ptr_wrapper!(Sampler -> SamplerInner);
pub struct Sampler(Arc<SamplerInner>);
impl Drop for SamplerInner {
    fn drop(&mut self) {
        unsafe { self.dev.dev.destroy_sampler(self.sampler, None) };
        info!("destroyed sampler");
    }
}
impl Sampler {
    fn create_sampler() -> Result<vk::Sampler> {
        unimplemented!();
    }
    fn new(dev: &Device, sampler_cfg: SamplerConfig) -> Result<Sampler> {
        let dev = dev.0.clone();
        unimplemented!();
    }
}


pub struct ShaderModuleInner {
    pub(crate) dev: Device,
    pub shader_mod: vk::ShaderModule,
    pub manifests: Vec<EntryPointManifest>,
}
impl_ptr_wrapper!(ShaderModule -> ShaderModuleInner);
pub struct ShaderModule(Arc<ShaderModuleInner>);
impl Drop for ShaderModuleInner {
    fn drop(&mut self) {
        unsafe { self.dev.dev.destroy_shader_module(self.shader_mod, None) };
        info!("destroyed shader module");
    }
}
impl ShaderModule {
    fn create_shader_mod(
        dev: &Device,
        spv: &SpirvBinary,
    ) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(spv.words())
            .build();
        let shader_mod = unsafe {
            dev.dev.create_shader_module(&create_info, None)?
        };
        info!("created shader module");
        Ok(shader_mod)
    }
    pub fn new(dev: &Device, spv: &SpirvBinary) -> Result<ShaderModule> {
        let dev = dev.clone();
        let shader_mod = Self::create_shader_mod(&dev, spv)?;
        let manifests = spv.reflect()?.to_vec();
        let inner = ShaderModuleInner { dev, shader_mod, manifests };
        Ok(ShaderModule(Arc::new(inner)))
    }
    pub fn entry_points(
        &self
    ) -> impl Iterator<Item=Result<ShaderEntryPoint>> + '_ {
        (0..self.manifests.len()).into_iter()
            .map(move |i| ShaderEntryPoint::new(self, i))
    }
}

// TODO: Separate the inner.
#[derive(Clone)]
pub struct ShaderEntryPoint {
    shader_mod: ShaderModule,
    ientry_point: usize,
}
impl ShaderEntryPoint {
    fn new(
        shader_mod: &ShaderModule,
        ientry_point: usize,
    ) -> Result<ShaderEntryPoint> {
        let shader_mod = shader_mod.clone();
        Ok(ShaderEntryPoint { shader_mod, ientry_point })
    }
    fn name(&self) -> &str {
        &self.manifest().name
    }
    fn stage(&self) -> Option<vk::ShaderStageFlags> {
        use spirq::ExecutionModel::*;
        let stage = match self.manifest().exec_model {
            Vertex => vk::ShaderStageFlags::VERTEX,
            TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            Geometry => vk::ShaderStageFlags::GEOMETRY,
            Fragment => vk::ShaderStageFlags::FRAGMENT,
            GLCompute => vk::ShaderStageFlags::COMPUTE,
            _ => { return None }
        };
        Some(stage)
    }
    pub fn manifest(&self) -> &EntryPointManifest {
        &self.shader_mod.manifests[self.ientry_point]
    }
}

fn spirq_desc_ty2vk_desc_ty(
    desc_ty: &spirq::ty::DescriptorType
) -> vk::DescriptorType {
    match desc_ty {
        DescriptorType::UniformBuffer(..) => {
            vk::DescriptorType::UNIFORM_BUFFER
        },
        DescriptorType::StorageBuffer(..) => {
            vk::DescriptorType::STORAGE_BUFFER
        },
        DescriptorType::Image(_, ty) => {
            if let spirq::ty::Type::Image(img_ty) = ty {
                if let spirq::ty::ImageUnitFormat::Sampled = img_ty.unit_fmt {
                    vk::DescriptorType::SAMPLED_IMAGE
                } else {
                    vk::DescriptorType::STORAGE_IMAGE
                }
            } else { unreachable!() }
        },
        DescriptorType::Sampler(..) => {
            vk::DescriptorType::SAMPLER
        }
        DescriptorType::SampledImage(..) => {
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        }
        DescriptorType::InputAttachment(..) => {
            vk::DescriptorType::INPUT_ATTACHMENT
        },
    }
}

pub struct ShaderArrayInner {
    entry_points: Vec<ShaderEntryPoint>,
    manifest: Manifest,
    /// `HashMap` for DescriptorUpdateTemplate.
    desc_set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    desc_pool_sizes: Vec<vk::DescriptorPoolSize>,
    push_const_rng: Vec<vk::PushConstantRange>,
}
impl Drop for ShaderArrayInner {
    fn drop(&mut self) {
        for &desc_set_layout in self.desc_set_layouts.values() {
            let dev = &self.entry_points.first().unwrap().shader_mod.dev;
            unsafe {
                dev.dev.destroy_descriptor_set_layout(desc_set_layout, None);
            }
        }
    }
}
impl_ptr_wrapper!(ShaderArray -> ShaderArrayInner);
pub struct ShaderArray(Arc<ShaderArrayInner>);
impl ShaderArray {
    /// `entry_points` should be sorted externally to represent the actual order
    /// of execution, otherwise the input and output interface would be
    /// corrupted.
    pub fn new(entry_points: &[ShaderEntryPoint]) -> Result<ShaderArray> {
        if entry_points.len() == 0 { return Err(Error::InvalidOperation) }
        let manifest = {
            let mut manifest = Manifest::default();
            for entry_point in entry_points {
                manifest.merge(entry_point.manifest());
            }
            manifest
        };
        let entry_points = entry_points.iter().cloned().collect::<Vec<_>>();
        let dev = &entry_points.first().unwrap().shader_mod.dev;

        let mut pool_size_map = HashMap::<vk::DescriptorType, u32>::new();
        let mut set_bind_map = HashMap::<u32, Vec<vk::DescriptorSetLayoutBinding>>::new();
        for desc in manifest.descs() {
            let (set, bind) = desc.desc_bind.into_inner();
            // Process actual descriptor bindings.
            let nbind = desc.desc_ty.nbind();
            let desc_ty = spirq_desc_ty2vk_desc_ty(&desc.desc_ty);
            let dslb = vk::DescriptorSetLayoutBinding::builder()
                .binding(bind)
                .descriptor_type(desc_ty)
                .descriptor_count(nbind)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build();
            set_bind_map.entry(set).or_default().push(dslb);
            *pool_size_map.entry(desc_ty).or_default() += 1;
        }
        let push_const_rng = manifest.get_push_const()
            .and_then(|x| x.nbyte())
            .map(|x| {
                vec![
                    vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::ALL,
                        offset: 0,
                        size: x as u32,
                    }
                ]
            })
            .unwrap_or_default();
        let desc_set_layouts = {
            let mut desc_set_layouts = HashMap::with_capacity(set_bind_map.len());
            for (set, dslbs) in set_bind_map {
                let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&dslbs)
                    .build();
                let desc_set_layout = unsafe {
                    dev.dev.create_descriptor_set_layout(&create_info, None)?
                };
                desc_set_layouts.insert(set, desc_set_layout);
            }
            desc_set_layouts
        };
        let desc_pool_sizes = {
            let mut desc_pool_sizes = Vec::with_capacity(pool_size_map.len());
            for (desc_ty, ndesc) in pool_size_map {
                let dps = vk::DescriptorPoolSize {
                    ty: desc_ty,
                    descriptor_count: ndesc,
                };
                desc_pool_sizes.push(dps);
            }
            desc_pool_sizes
        };

        let inner = ShaderArrayInner {
            entry_points, manifest, desc_set_layouts, desc_pool_sizes,
            push_const_rng
        };
        Ok(ShaderArray(Arc::new(inner)))
    }
}





















#[derive(Clone)]
pub struct AttributeBinding {
    /// Vertex input binding.
    pub bind: u32,
    /// Offset from the bound vertex buffer.
    pub offset: usize,
    /// Stride to iterate the bound vertex buffer.
    pub stride: usize,
    /// Format of attribute units.
    pub fmt: vk::Format,
}
pub trait VertexHead {
    /// Get the binding point of a attribute.
    fn attr_bind(&self, loc: u32) -> Option<&AttributeBinding>;
}

#[derive(Clone)]
pub struct AttachmentReference {
    /// The index of attachment in the framebuffer. For shaders using input
    /// attachments, the input attachment indices MUST point to the indices of
    /// corresponding attachments in the framebuffer.
    pub attm_idx: u32,
    /// Format of attachment units. It will be deduced here whether the
    /// attachment is a depth-stencil attachment.
    pub fmt: vk::Format,
    /// The operation done on loading.
    ///
    /// WARNING: Currently CLEAR variant is not supported.
    pub load_op: vk::AttachmentLoadOp,
    /// The operation done on storing.
    pub store_op: vk::AttachmentStoreOp,
    /// Blend state of the attachment. Any non-`None` value will enable
    /// attachment blending.
    pub blend_state: Option<vk::PipelineColorBlendAttachmentState>,
    /// Final layout after attachment write.
    pub final_layout: vk::ImageLayout,
}
pub trait FragmentHead {
    fn attm_ref(&self, loc: u32) -> Option<&AttachmentReference>;
    fn depth_attm_idx(&self, depth_attm_id: u32) -> Option<u32>;
}

#[derive(Default)]
pub struct GraphicsDepthStencilConfig {
    /// Compare operator used in depth test.
    cmp_op: vk::CompareOp,
    /// Any non-`None` value in this field will enable depth bounds testing.
    /// Primitives are only drawn if they falls in this range of depth, where
    /// the first value if the minimal allowed depth and the second the maximum.
    depth_range: Option<(f32, f32)>,
    /// Any non-`None` value in this field will enable stencil testing. The
    /// first value is the front op state, and the second the back op state.
    stencil_ops: Option<(vk::StencilOpState, vk::StencilOpState)>,
}
#[derive(Default)]
pub struct GraphicsRasterizationConfig {
    pub wireframe: bool,
    pub cull_mode: vk::CullModeFlags,
}
#[derive(Default)]
pub struct GraphicsColorBlendConfig {
    pub src_factor: vk::BlendFactor,
    pub dst_factor: vk::BlendFactor,
    pub blend_op: vk::BlendOp,
    pub blend_consts: [f32; 4],
}
pub struct GraphicsPipeline {
    pub shader_arr: Arc<ShaderArrayInner>,
    pub attr_map: HashMap<InterfaceLocation, AttributeBinding>,
    pub attm_map: HashMap<InterfaceLocation, AttachmentReference>,
    /// Index of the depth-stencil attachment to be written in the framebuffer.
    pub depth_attm_idx: Option<u32>,
    pub raster_cfg: GraphicsRasterizationConfig,
    /// Any non-`None` value will enable depth testing.
    pub depth_cfg: Option<GraphicsDepthStencilConfig>,
    /// Any non-`None` value will enable alpha blending. All attachments will
    /// share the same blending configuration.
    pub blend_cfg: Option<GraphicsColorBlendConfig>,
}
impl GraphicsPipeline {
    /// Note that `depth_attm_id` is NOT the index of attachment in framebuffer
    /// but identifier to be used to query the provided `frag_head`.
    pub fn new<VH: VertexHead, FH: FragmentHead>(
        shader_arr: &ShaderArray,
        vert_head: &VH,
        frag_head: &FH,
        depth_attm_id: Option<u32>,
        raster_cfg: GraphicsRasterizationConfig,
        depth_cfg: Option<GraphicsDepthStencilConfig>,
        blend_cfg: Option<GraphicsColorBlendConfig>,
    ) -> Result<GraphicsPipeline> {
        // Ensure the vertex shader is the first and the fragment shader the
        // last.
        let manifest = &shader_arr.manifest;
        let attr_map = Self::collect_attr_map(manifest, vert_head)?;
        let attm_map = Self::collect_attm_map(manifest, frag_head)?;
        let depth_attm_idx = if let Some(depth_attm_id) = depth_attm_id {
            let depth_attm_idx = frag_head.depth_attm_idx(depth_attm_id)
                .ok_or(Error::InvalidOperation)?;
            Some(depth_attm_idx)
        } else { None };
        let shader_arr = shader_arr.0.clone();
        let graph_pipe = GraphicsPipeline {
            shader_arr, attr_map, attm_map, depth_attm_idx, raster_cfg,
            depth_cfg, blend_cfg
        };
        Ok(graph_pipe)
    }
    fn collect_attr_map(
        manifest: &Manifest,
        vert_head: &dyn VertexHead,
    ) -> Result<HashMap<InterfaceLocation, AttributeBinding>> {
        use std::collections::hash_map::Entry::Vacant;
        let mut attr_map = HashMap::new();
        for attr_res in manifest.inputs() {
            let location = attr_res.location;
            if let None = attr_res.ty.nbyte() {
                return Err(Error::PipelineMismatched("attachment cannot be opaque type"));
            }
            let attr_bind = vert_head.attr_bind(location.loc())
                // Attribute not supported.
                .ok_or(Error::PipelineMismatched("attribute not supported by head"))?;
            if let Vacant(entry) = attr_map.entry(location) {
                entry.insert(attr_bind.clone());
            } else {
                return Err(Error::PipelineMismatched("attribute location collision"));
            }
        }
        Ok(attr_map)
    }
    fn collect_attm_map(
        manifest: &Manifest,
        frag_head: &dyn FragmentHead,
    ) -> Result<HashMap<InterfaceLocation, AttachmentReference>> {
        use std::collections::hash_map::Entry::Vacant;
        let mut attm_map = HashMap::new();
        for attm_ref in manifest.outputs() {
            let location = attm_ref.location;
            if let None = attm_ref.ty.nbyte() {
                return Err(Error::PipelineMismatched("attachment cannot be opaque type"));
            }
            let attm_ref = frag_head.attm_ref(location.loc())
                .ok_or(Error::PipelineMismatched("attachment not supported by head"))?;
            if let Vacant(entry) = attm_map.entry(location) {
                entry.insert(attm_ref.clone());
            } else {
                return Err(Error::PipelineMismatched("attachment location collision"));
            }
        }
        Ok(attm_map)
    }
}


fn is_depth_fmt(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM |
        vk::Format::X8_D24_UNORM_PACK32 |
        vk::Format::D32_SFLOAT |
        vk::Format::S8_UINT |
        vk::Format::D16_UNORM_S8_UINT |
        vk::Format::D24_UNORM_S8_UINT |
        vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}


pub struct ComputePipeline {
    pub shader_arr: Arc<ShaderArrayInner>,
}
impl ComputePipeline {
    /// Create a new compute pipeline configuration. `shader_arr` MUST consist
    /// of one and only one compute shader.
    pub fn new(shader_arr: &ShaderArray) -> Result<ComputePipeline> {
        let shader_arr = shader_arr.0.clone();
        let comp_pipe = ComputePipeline { shader_arr };

        Ok(comp_pipe)
    }
}


struct PipelineInner {
    dev: Arc<DeviceInner>,
    shader_arr: Arc<ShaderArrayInner>,
    pipe: vk::Pipeline,
    pipe_layout: vk::PipelineLayout,
    pipe_bp: vk::PipelineBindPoint,
}
impl Drop for PipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.dev.dev.destroy_pipeline(self.pipe, None);
            self.dev.dev.destroy_pipeline_layout(self.pipe_layout, None);
        }
    }
}
impl_ptr_wrapper!(Pipeline -> PipelineInner);
struct Pipeline(Arc<PipelineInner>);
impl Pipeline {
    fn new(
        dev: Arc<DeviceInner>,
        shader_arr: Arc<ShaderArrayInner>,
        pipe: vk::Pipeline,
        pipe_layout: vk::PipelineLayout,
        pipe_bp: vk::PipelineBindPoint,
    ) -> Pipeline {
        let inner = PipelineInner {
            dev, shader_arr, pipe, pipe_layout, pipe_bp
        };
        Pipeline(Arc::new(inner))
    }
}


pub struct RenderPassInner {
    dev: Arc<DeviceInner>,
    pass: vk::RenderPass,
    pipes: Vec<Pipeline>,
    // (w, h)
    framebuf_extent: (u32, u32),
}
impl Drop for RenderPassInner {
    fn drop(&mut self) {
        let dev = &self.dev.dev;
        unsafe { dev.destroy_render_pass(self.pass, None); }
        for pipe in self.pipes.iter() {
            unsafe { dev.destroy_pipeline(pipe.pipe, None); }
        }
    }
}
impl_ptr_wrapper!(RenderPass -> RenderPassInner);
pub struct RenderPass(Arc<RenderPassInner>);
impl RenderPass {
    fn create_pass(
        dev: &DeviceInner,
        graph_pipes: &[GraphicsPipeline],
        deps: &[vk::SubpassDependency],
    ) -> Result<vk::RenderPass> {
        // WARNING: Attachments indices from pipelines MUST NOT overlap.
        let attm_ref_map = graph_pipes.iter()
            .flat_map(|graph_pipe| {
                graph_pipe.attm_map.values()
                    .map(|attm_ref| (attm_ref.attm_idx, attm_ref))
            })
            .collect::<HashMap<_, _>>();
        let attm_descs = attm_ref_map.iter()
            .map(|(_attm_idx, attm_ref)| {
                vk::AttachmentDescription {
                    flags: Default::default(),
                    format: attm_ref.fmt,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: attm_ref.load_op,
                    store_op: attm_ref.store_op,
                    stencil_load_op: attm_ref.load_op,
                    stencil_store_op: attm_ref.store_op,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: attm_ref.final_layout,
                }
            })
            .collect::<Vec<_>>();
        struct SubpassDescriptionDetail {
            input_attms: Vec<vk::AttachmentReference>,
            color_attms: Vec<vk::AttachmentReference>,
            depth_attm: Option<vk::AttachmentReference>,
        }
        let mut subpass_desc_details = Vec::with_capacity(graph_pipes.len());
        for graph_pipe in graph_pipes.iter() {
            let mut input_attms = Vec::new();
            for desc_res in graph_pipe.shader_arr.manifest.descs() {
                if let DescriptorType::InputAttachment(nbind, idx) = desc_res.desc_ty {
                    let (nbind, idx) = (*nbind, *idx);
                    let len = (idx + nbind) as usize;
                    input_attms.reserve(len);
                    while input_attms.len() < len {
                        let attm_ref = vk::AttachmentReference {
                            attachment: vk::ATTACHMENT_UNUSED,
                            layout: Default::default(),
                        };
                        input_attms.push(attm_ref);
                    }
                    let attm_fmt = attm_ref_map.get(&idx)
                        .ok_or(Error::InvalidOperation)?
                        .fmt;
                    let layout = if is_depth_fmt(attm_fmt) {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    };
                    for j in idx..(idx + nbind) {
                        let attm_ref = vk::AttachmentReference {
                            attachment: j,
                            layout,
                        };
                        input_attms[j as usize] = attm_ref;
                    }
                }
            }
            let mut color_attms = Vec::new();
            for (location, attm_ref) in graph_pipe.attm_map.iter() {
                let loc = location.loc() as usize;
                if color_attms.len() < loc {
                    let stuff = (color_attms.len()..loc).into_iter()
                        .map(|_| {
                            vk::AttachmentReference {
                                attachment: vk::ATTACHMENT_UNUSED,
                                layout: Default::default(),
                            }
                        });
                    color_attms.extend(stuff);
                }
                let attm_ref = vk::AttachmentReference {
                    attachment: attm_ref.attm_idx,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                };
                color_attms.push(attm_ref);
            }
            let depth_attm = graph_pipe.depth_attm_idx
                .map(|depth_attm_idx| {
                    vk::AttachmentReference {
                        attachment: depth_attm_idx,
                        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    }
                });
            let subpass_desc_detail = SubpassDescriptionDetail {
                color_attms, input_attms, depth_attm
            };
            subpass_desc_details.push(subpass_desc_detail);
        };
        
        let subpass_descs = subpass_desc_details.iter()
            .map(|subpass_desc_detail| {
                let mut x = vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&subpass_desc_detail.color_attms)
                    .input_attachments(&subpass_desc_detail.input_attms);
                if let Some(ref depth_attm) = subpass_desc_detail.depth_attm {
                    x = x.depth_stencil_attachment(depth_attm);
                }
                x.build()
            })
            .collect::<Vec<_>>();

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attm_descs)
            .subpasses(&subpass_descs)
            .dependencies(deps)
            .build();
        let pass = unsafe {
            dev.dev.create_render_pass(&create_info, None)?
        };
        Ok(pass)
    }
    fn create_graph_pipes(
        dev: Arc<DeviceInner>,
        graph_pipes: &[GraphicsPipeline],
        pass: vk::RenderPass,
        framebuf_extent: (u32, u32),
    ) -> Result<Vec<Pipeline>> {
        struct GraphicsPipelineDescriptionDetail {
            entry_names: Vec<std::ffi::CString>,
            psscis: Vec<vk::PipelineShaderStageCreateInfo>,
            vert_binds: Vec<vk::VertexInputBindingDescription>,
            vert_attrs: Vec<vk::VertexInputAttributeDescription>,
            pvisci: vk::PipelineVertexInputStateCreateInfo,
            piasci: vk::PipelineInputAssemblyStateCreateInfo,
            ptsci: vk::PipelineTessellationStateCreateInfo,
            pvsci: vk::PipelineViewportStateCreateInfo,
            prsci: vk::PipelineRasterizationStateCreateInfo,
            pmsci: vk::PipelineMultisampleStateCreateInfo,
            pdssci: vk::PipelineDepthStencilStateCreateInfo,
            attm_blends: Vec<vk::PipelineColorBlendAttachmentState>,
            pcbsci: vk::PipelineColorBlendStateCreateInfo,
            pdsci: vk::PipelineDynamicStateCreateInfo,
            pipe_layout: vk::PipelineLayout,
        }
        // WARNING: The following code should not break and if we are
        // introducing any optimization, PLEASE CHECK THAT NO DANGLING POINTER
        // FORMS.

        let (width, height) = framebuf_extent;
        let viewports = &[
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        ];
        let scissors = &[
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
        ];

        let mut graph_pipe_desc_details = Vec::new();
        for graph_pipe in graph_pipes {
            let shader_arr = graph_pipe.shader_arr.clone();
            let entry_names = shader_arr.entry_points.iter()
                .map(|entry_point| {
                    std::ffi::CString::new(entry_point.name())
                        .expect("invalid entry point name")
                })
                .collect::<Vec<_>>();
            let psscis = shader_arr.entry_points.iter()
                .enumerate()
                .map(|(i, entry_point)| {
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(entry_point.stage().expect("invalid stage"))
                        .module(entry_point.shader_mod.shader_mod)
                        // .specialization_info(/* ... */) // TODO
                        .name(&entry_names[i])
                        .build()
                })
                .collect::<Vec<_>>();
            let mut vert_bind_map = HashMap::new();
            let mut vert_attr_map = HashMap::new();
            for (location, attr_bind) in graph_pipe.attr_map.iter() {
                use std::collections::hash_map::Entry::Vacant;
                if let Vacant(entry) = vert_bind_map.entry(attr_bind.bind) {
                    let vert_bind = vk::VertexInputBindingDescription {
                        binding: attr_bind.bind,
                        stride: attr_bind.stride as u32,
                        input_rate: vk::VertexInputRate::VERTEX,
                    };
                    entry.insert(vert_bind);
                }
                if let Vacant(entry) = vert_attr_map.entry(*location) {
                    let vert_attr = vk::VertexInputAttributeDescription {
                        location: location.loc(),
                        binding: attr_bind.bind,
                        format: attr_bind.fmt,
                        offset: attr_bind.offset as u32,
                    };
                    entry.insert(vert_attr);
                }
            }
            let vert_binds = vert_bind_map.into_iter()
                .map(|x| x.1)
                .collect::<Vec<_>>();
            let vert_attrs = vert_attr_map.into_iter()
                .map(|x| x.1)
                .collect::<Vec<_>>();
            let pvisci = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&vert_binds)
                .vertex_attribute_descriptions(&vert_attrs)
                .build();

            
            let piasci = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                // TODO: This one is interesting. Wanna look into it?
                .primitive_restart_enable(false)
                .build();

            // NOTE: Don't use tesselation and geometry shaders because
            // these stages will generate unknown number of primitives,
            // which will disable tile-based rendering which has been used
            // broadly.
            let ptsci = vk::PipelineTessellationStateCreateInfo::builder()
                .build();

            let pvsci = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(viewports.len() as u32)
                .viewports(viewports)
                .scissor_count(scissors.len() as u32)
                .scissors(scissors)
                .build();

            let poly_mode = if graph_pipe.raster_cfg.wireframe {
                vk::PolygonMode::LINE
            } else {
                vk::PolygonMode::FILL
            };
            let prsci = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(poly_mode)
                .cull_mode(graph_pipe.raster_cfg.cull_mode)
                .depth_bias_enable(false)
                .line_width(1.0)
                .build();

            let pmsci = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .sample_shading_enable(false)
                .min_sample_shading(1.0)
                .build();

            let mut pdssci = vk::PipelineDepthStencilStateCreateInfo::builder()
                .build();
            if let Some(depth_cfg) = graph_pipe.depth_cfg.as_ref() {
                pdssci.depth_test_enable = 1;
                pdssci.depth_compare_op = depth_cfg.cmp_op;
                // This will be ignored if there is no depth-stencil
                // attachment attached.
                pdssci.depth_write_enable = 1;
                if dev.cap_detail.feats.depth_bounds != 0 {
                    if let Some((min_depth, max_depth)) = depth_cfg.depth_range.as_ref() {
                        pdssci.depth_bounds_test_enable = 1;
                        pdssci.min_depth_bounds = *min_depth;
                        pdssci.max_depth_bounds = *max_depth;
                    }
                }
                if let Some((front_op, back_op)) = depth_cfg.stencil_ops.as_ref() {
                    pdssci.stencil_test_enable = 1;
                    pdssci.front = *front_op;
                    pdssci.back = *back_op;
                }
            }

            let (attm_blend, blend_consts) = {
                if let Some(blend_cfg) = graph_pipe.blend_cfg.as_ref() {
                    let ab = vk::PipelineColorBlendAttachmentState::builder()
                        .blend_enable(true)
                        .src_color_blend_factor(blend_cfg.src_factor)
                        .dst_color_blend_factor(blend_cfg.dst_factor)
                        .color_blend_op(blend_cfg.blend_op)
                        .src_alpha_blend_factor(blend_cfg.src_factor)
                        .dst_alpha_blend_factor(blend_cfg.dst_factor)
                        .alpha_blend_op(blend_cfg.blend_op)
                        .color_write_mask(vk::ColorComponentFlags::all())
                        .build()
                    (ab, blend_cfg.blend_consts)
                } else {
                    vk::PipelineColorBlendAttachmentState::builder()
                        .blend_enable(false)
                        .build()
                    (ab, Default::default())
                }
            };
            let mut attm_blends = std::iter::repeat(attm_blend)
                .take(graph_pipe.attm_map.len())
                .collect::<Vec<_>>();

            let pcbsci = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&attm_blends)
                .blend_constants(blend_consts)
                .build();

            let pdsci = vk::PipelineDynamicStateCreateInfo::builder()
                .build();

            let desc_set_layouts = &shader_arr.desc_set_layouts;
            let desc_set_layouts = desc_set_layouts.values()
                .copied()
                .collect::<Vec<_>>();
            let create_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&desc_set_layouts)
                .push_constant_ranges(&shader_arr.push_const_rng)
                .build();
            let pipe_layout = unsafe {
                dev.dev.create_pipeline_layout(&create_info, None)?
            };

            let graph_pipe_desc_detail = GraphicsPipelineDescriptionDetail {
                entry_names, psscis, vert_binds, vert_attrs, pvisci, piasci,
                ptsci, pvsci, prsci, pmsci, pdssci, attm_blends, pcbsci,
                pdsci, pipe_layout
            };
            graph_pipe_desc_details.push(graph_pipe_desc_detail);
        }

        let create_infos = graph_pipe_desc_details.iter()
            .enumerate()
            .map(|(i, graph_pipe_desc_detail)| {
                vk::GraphicsPipelineCreateInfo::builder()
                    .stages(&graph_pipe_desc_detail.psscis)
                    .vertex_input_state(&graph_pipe_desc_detail.pvisci)
                    .input_assembly_state(&graph_pipe_desc_detail.piasci)
                    .tessellation_state(&graph_pipe_desc_detail.ptsci)
                    .viewport_state(&graph_pipe_desc_detail.pvsci)
                    .rasterization_state(&graph_pipe_desc_detail.prsci)
                    .multisample_state(&graph_pipe_desc_detail.pmsci)
                    .depth_stencil_state(&graph_pipe_desc_detail.pdssci)
                    .color_blend_state(&graph_pipe_desc_detail.pcbsci)
                    .dynamic_state(&graph_pipe_desc_detail.pdsci)
                    .layout(graph_pipe_desc_detail.pipe_layout)
                    .render_pass(pass)
                    .subpass(i as u32)
                    .build()
            })
            .collect::<Vec<_>>();
        let pipe_cache = vk::PipelineCache::null();
        let pipes = unsafe {
            dev.dev.create_graphics_pipelines(pipe_cache, &create_infos, None)
        };
        let pipes = match pipes {
            Ok(pipes) => pipes,
            Err((pipes, e)) => {
                for pipe in pipes {
                    if pipe != vk::Pipeline::null() {
                        unsafe { dev.dev.destroy_pipeline(pipe, None) };
                    }
                }
                return Err(e.into());
            }
        };
        let pipes = pipes.into_iter()
            .enumerate()
            .map(|(i, pipe)| {
                let pipe_layout = graph_pipe_desc_details[i].pipe_layout;
                Pipeline::new(dev.clone(), graph_pipes[i].shader_arr.clone(),
                    pipe, pipe_layout, vk::PipelineBindPoint::GRAPHICS)
            })
            .collect::<Vec<_>>();
        Ok(pipes)
    }
    /// `graph_pipes` should be sorted in order to represent the command
    /// recording order.
    pub fn new(
        dev: &Device,
        graph_pipes: &[GraphicsPipeline],
        deps: &[vk::SubpassDependency],
        framebuf_extent: (u32, u32),
    ) -> Result<RenderPass> {
        let dev = dev.0.clone();
        let pass = Self::create_pass(&dev, graph_pipes, deps)?;
        let pipes = Self::create_graph_pipes(
            dev.clone(), graph_pipes, pass, framebuf_extent
        )?;

        let inner = RenderPassInner { dev, pass, pipes, framebuf_extent };
        Ok(RenderPass(Arc::new(inner)))
    }
}


struct ComputeTaskInner {
    dev: Arc<DeviceInner>,
    pipes: Vec<Pipeline>,
}
impl Drop for ComputeTaskInner {
    fn drop(&mut self) {
        let dev = &self.dev.dev;
        for pipe in self.pipes.iter() {
            unsafe { dev.destroy_pipeline(pipe.pipe, None); }
        }
    }
}
impl_ptr_wrapper!(ComputeTask -> ComputeTaskInner);
struct ComputeTask(Arc<ComputeTaskInner>);
impl ComputeTask {
    fn create_comp_pipes(
        dev: &ash::Device,
        comp_pipes: &[ComputePipeline],
    ) -> Result<Vec<Pipeline>> {
        unimplemented!();
        /*
        for comp_pipe in comp_pipes {
            let entry_point = comp_pipe.shader_arr.entry_points.first();
            let entry_name = std::ffi::CString::new(entry_point.name())
                .expect("invalid entry point name");
            let pssci = vk::PipelineShaderStageCreateInfo::builder()
                .stage(entry_point.stage().expect("invalid stage"))
                .module(entry_point.shader_mod.shader_mod)
                // .specialization_info(/* ... */) // TODO
                .name(&entry_name)
                .build();
                
            let desc_set_layouts = &comp_pipe.shader_arr.desc_set_layouts;
            let desc_set_layouts = desc_set_layouts.values()
                .copied()
                .collect::<Vec<_>>();
            let create_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&desc_set_layouts)
                .push_constant_ranges(&comp_pipe.shader_arr.push_const_rng)
                .build();
            let pipe_layout = unsafe {
                dev.0.dev.create_pipeline_layout(&create_info, None)?
            };
            let create_info = vk::ComputePipelineCreateInfo::builder()
                .stage(pssci)
                .layout(pipe_layout)
                .build();
            let pipe = 
        }
        */
    }
    /// Create a new compute task. A compute task consists of one or multiple
    /// compute pipelines executed sequentially.
    pub fn new(
        dev: &Device,
        comp_pipes: &[ComputePipeline],
    ) -> Result<ComputeTask> {
        let dev = dev.0.clone();
        let pipes = Self::create_comp_pipes(
            &dev.dev, comp_pipes
        )?;

        let inner = ComputeTaskInner { dev, pipes };
        Ok(ComputeTask(Arc::new(inner)))
    }
}






#[derive(Clone, Copy)]
pub enum BindPoint {
    /// Bind a resource to a descriptor binding point. Descriptor set will be
    /// internally managed by `FlowState`. Descriptor's type will be
    /// automatically inferred shader module metadata. The binding will be
    /// visible to all following `draw` calls.
    ///
    /// NOTE: Bound variable MUST be `Buffer` or `Image`.
    Descriptor(u32, u32),
    /// Bind a vertex buffer for graphics pipeline. The binding will be visible
    /// to all following `draw` calls.
    ///
    /// NOTE: Bound variable must be `Buffer`.
    VertexInput(u32),
    /// Bind an index buffer for graphics pipeline. The binding will be visible
    /// to all following `draw` calls.
    ///
    /// NOTE: Bound variable must be `Buffer`.
    Index,
    /// Bind an color attachment to the framebuffer at the given index. Here the
    /// term attachment is a general idea which can be depth/stencil buffer or
    /// color buffer.
    ///
    /// NOTE: Bound variable MUST be `Image`.
    Attachment(u32),
}

type VariableToken = usize;
#[derive(Clone)]
struct TransferEventArgs {
    src_var_idx: VariableToken,
    dst_var_idx: VariableToken,
}
#[derive(Clone)]
struct PushConstantEventArgs {
    push_const_var_idx: VariableToken,
}
#[derive(Clone)]
struct BindEventArgs {
    bp: BindPoint,
    var_idx: Option<VariableToken>,
}
#[derive(Clone)]
struct DispatchEventArgs {
    task: Arc<ComputeTaskInner>,
    nworkgrp_x_var_idx: VariableToken,
    nworkgrp_y_var_idx: VariableToken,
    nworkgrp_z_var_idx: VariableToken,
}
#[derive(Clone)]
struct DrawEventArgs {
    pass: Arc<RenderPassInner>,
    nvert_var_idx: VariableToken,
    ninst_var_idx: VariableToken,
}
#[derive(Clone)]
enum Event {
    Transfer(TransferEventArgs),
    PushConstant(PushConstantEventArgs),
    Bind(BindEventArgs),
    Dispatch(DispatchEventArgs),
    Draw(DrawEventArgs),
}
impl Event {
    /// Get the pipeline binding point of the current event. For example,
    /// compute dispatches need to be bound to compute pipeline; but transfers
    /// doesn't need a pipeline so it has no binding point.
    fn pipe_bind(&self) -> Option<vk::PipelineBindPoint> {
        match self {
            Self::Dispatch(_) => Some(vk::PipelineBindPoint::COMPUTE),
            Self::Draw(_) => Some(vk::PipelineBindPoint::GRAPHICS),
            _ => None,
        }
    }
    /// Get the queue interface the event should be submitted to.
    fn qi(&self) -> Option<QueueInterface> {
        match self {
            Self::Transfer(_) => Some(QueueInterface::Transfer),
            Self::Dispatch(_) => Some(QueueInterface::Compute),
            Self::Draw(_) => Some(QueueInterface::Graphics),
            _ => None,
        }
    }
}


struct Chunk {
    /// Serials of dependencies. Events are not triggered *until* the dependency
    /// has finished.
    ///
    /// This dependency is `FlowHead` dependency, not `Chunk` dependency. This
    /// `FlowHead` dependency will be then converted to `Chunk` dependency
    /// later, tho.
    deps: Vec<usize>,
    events: Vec<Event>,
    /// The queue interface the chunk has been bound to. It depends on the last
    /// interface-bound event in this chunk. A chunk only contains events bound
    /// to a same interface.
    qi: Option<QueueInterface>,
}
pub struct FlowHead {
    serial: usize,
}
pub struct Flow {
    serial: usize,
    chunks: Vec<Chunk>,
}
impl Flow {
    fn new(serial: usize) -> Self {
        let chunk = Chunk {
            deps: Vec::new(),
            events: Vec::new(),
            qi: None,
        };
        let chunks = vec![chunk];
        Flow { serial, chunks }
    }


    fn last_chunk(&mut self) -> &mut Chunk {
        if let Some(chunk) = self.chunks.last_mut() {
            return chunk;
        } else {
            unreachable!("flow head always have one chunk ready");
        }
    }
    /// Wait on an previous flow head. Waiting on a flow head itself is
    /// considered no-op.
    pub fn wait_on(&mut self, head: &FlowHead) -> &mut Self {
        // Waiting on self is no-op.
        if head.serial == self.serial {
            return self;
        }
        let last_chunk = self.last_chunk();
        if last_chunk.events.is_empty() {
            // There is no event triggered since last wait, so the new wait can
            // be batched with the previous one.
            last_chunk.deps.push(head.serial);
        } else {
            // There are events triggered before, so we split and add a new
            // chunk.
            let chunk = Chunk {
                deps: vec![head.serial],
                events: Vec::new(),
                qi: None,
            };
            self.chunks.push(chunk);
        }
        self
    }
    
    fn push_event(&mut self, event: Event) -> &mut Self {
        if let Some(qi) = event.qi() {
            let mut last_chunk = self.last_chunk();
            if let Some(last_qi) = last_chunk.qi {
                // The interface has already been assigned.
                if last_qi != qi {
                    // Split and add a new chunk if the queue interface
                    // requirements are in conflict.
                    let ilast_bound = last_chunk.events.iter()
                        .rposition(|x| x.qi().is_some())
                        .expect("queue interface not assigned for events but for chunks");
                    let events = last_chunk.events
                        .drain((ilast_bound + 1)..)
                        .collect();
                    let chunk = Chunk {
                        deps: Vec::new(),
                        events,
                        qi: Some(qi),
                    };
                    self.chunks.push(chunk);
                }
            } else {
                // The interface has not been assigned yet. Set the current
                // event's interface as chunk interface.
                last_chunk.qi = Some(qi);
            }
        }
        self.last_chunk().events.push(event);
        self
    }
    /// Copy the data contained in `src` to `dst`. Staging buffer might be
    /// lazily allocated for inter-chip data transfer if either of the memory is
    /// not host visible. Due to pixel packing, images are always transfered via
    /// a staging buffer.
    ///
    /// NOTE: `src` and `dst` MUST be `HostMemory`, `Buffer` or `Image`.
    pub fn transfer(&mut self, src: VariableToken, dst: VariableToken) -> &mut Self {
        let event = Event::Transfer(TransferEventArgs {
            src_var_idx: src,
            dst_var_idx: dst,
        });
        self.push_event(event);
        self
    }
    /// Push constant data as a part of the control flow to the device.
    ///
    /// NOTE: `push_const` MUST be `HostMemory`.
    pub fn push_const(&mut self, push_const: VariableToken) -> &mut Self {
        let event = Event::PushConstant(PushConstantEventArgs {
            push_const_var_idx: push_const,
        });
        self.push_event(event);
        self
    }
    /// Bind a variable to the binding point.
    ///
    /// NOTE: `var` MUST conform the requirements as described in `BindPoint`
    /// documentation.
    pub fn bind(&mut self, bp: BindPoint, var: Option<VariableToken>) -> &mut Self {
        let event = Event::Bind(BindEventArgs { bp, var_idx: var });
        self.push_event(event);
        self
    }
    /// Dispatch `nworkgrp` workgroups, factorized to optimal dimensions.
    ///
    /// NOTE: `nworkgrp` MUST be `Count`.
    // fn dispatch(&mut self, comp_pipe: &ComputePipeline, nworkgrp: VariableToken) -> &mut Self;
    /// If there is index buffer bound to the pipeline, vertex elements are
    /// indexed by the indices specified in the index buffer.
    pub fn draw(&mut self, pass: &RenderPass, nvert: VariableToken, ninst: VariableToken) -> &mut Self {
        let event = Event::Draw(DrawEventArgs {
            pass: pass.0.clone(),
            nvert_var_idx: nvert,
            ninst_var_idx: ninst,
        });
        self.push_event(event);
        self
    }


    pub fn pause(&self) -> FlowHead {
        FlowHead { serial: self.serial }
    }
}
pub struct FlowGraph {
    /// Chunks of events ordered in a way that all chunks will not depend on any
    /// chunk which has higher index than it.
    chunks: Vec<Chunk>,
    /// Reversed dependency mapping from dependee to dependers. Note that the
    /// length of this list is the number of semaphores needed for one
    /// execution; the length of this list plus one is the number of command
    /// buffer needed for one execution of the flow graph.
    rev_dep_map: Vec<Vec<usize>>,
}
impl FlowGraph {
    fn new(head: &FlowHead, flows: &[Flow]) -> FlowGraph {
        let chunks = Self::flatten_flow_refs(head, flows);
        let rev_dep_map = Self::rev_deps(&chunks);
        FlowGraph { chunks, rev_dep_map }
    }
    fn flatten_flow_refs(
        head: &FlowHead,
        flows: &[Flow],
    ) -> Vec<Chunk> {
        fn fn_impl<'a>(
            flow_serial: usize,
            flows: &[Flow],
            chunks: &mut Vec<Chunk>,
            // Mapping from flow serial to a range of `chunks`.
            rng_map: &mut [Option<Range<usize>>],
        ) {
            let beg = chunks.len();
            for (i, chunk) in flows[flow_serial].chunks.iter().enumerate() {
                // Recursively resolve dependency, ensure chunks are already
                // appended before we refer to them.
                for &dep in chunk.deps.iter() {
                    if let None = rng_map[dep] {
                        fn_impl(flow_serial, flows, chunks, rng_map);
                    }
                }
                // Don't replace the `i` below with `chunks.deps.len()`.
                // Remember that `chunks` has been modified.
                let nexdep = chunk.deps.len();
                let mut deps = if i == 0 {
                    // No internal reference for the first chunk.
                    Vec::with_capacity(nexdep)
                } else {
                    // Add internal reference first.
                    let mut deps = Vec::with_capacity(nexdep + 1);
                    deps.push(beg + i - 1);
                    deps
                };
                // Append external reference. Convert flow dependency to chunk
                // dependency.
                let exref = chunk.deps.iter()
                    .map(|&flow_serial| {
                        let rng = rng_map[flow_serial].as_ref().unwrap();
                        rng.end - 1
                    });
                deps.extend(exref);
    
                let events = chunk.events.clone();
                chunks.push(Chunk { deps, events, qi: None });
            }
            let end = chunks.len();
            rng_map[flow_serial] = Some(beg..end);
        }
        // All chunks flatted from head.
        let mut chunks = Vec::new();
        // The range of chunks included in a `Flow` in `chunks`.
        let mut rng_map = std::iter::repeat(None)
            .take(flows.len())
            .collect::<Vec<_>>();

        fn_impl(head.serial, &flows, &mut chunks, &mut rng_map);
        // We won't need the flow head.
        chunks
    }
    fn rev_deps(chunks: &[Chunk]) -> Vec<Vec<usize>> {
        // Revert the dependency relation.
        // Mapping from chunk index to the indices of whom depends on the chunk.
        let mut rev_dep_map = std::iter::repeat(Vec::new())
            .take(chunks.len())
            .collect::<Vec<_>>();
        for (i, chunk) in chunks.iter().enumerate() {
            for dep in chunk.deps.iter() {
                rev_dep_map[*dep].push(i);
            }
        }
        rev_dep_map
    }
}


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    HostMemory,
    Buffer,
    Image,
    Sampler,
    SampledImage,
    Count,
}

pub struct SymbolSource {
    /// Flows.
    flows: Vec<Flow>,
    /// Variable tokens are mapped to indices of `syms`.
    syms: Vec<VariableType>,
    /// Mapping from symbol names to symbol tokens. One-to-one mapping.
    name_map: HashMap<String, VariableToken>,
}
impl SymbolSource {
    fn new() -> SymbolSource {
        SymbolSource {
            flows: Vec::new(),
            syms: Vec::new(),
            name_map: HashMap::new(),
        }
    }

    pub fn flow(&mut self) -> &mut Flow {
        let serial = self.flows.len();
        self.flows.push(Flow::new(serial));
        &mut self.flows[serial]
    }

    fn push_sym(&mut self, name: &str, ty: VariableType) -> VariableToken {
        let token = self.syms.len();
        self.syms.push(ty);
        self.name_map.insert(name.to_owned(), token);
        token
    }
    /// Declare a piece of unstructured host memory. Push constants and memory
    /// transfer from host will need this type of variables.
    pub fn host_mem(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::HostMemory)
    }
    /// Declare a buffer variable.
    pub fn buf(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Buffer)
    }
    /// Declare a image variable.
    pub fn img(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Image)
    }
    /// Declare a separable sampler.
    pub fn sampler(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Sampler)
    }
    /// Delcare a integral number. Compute pipeline's work group size and
    /// graphics pipeline's vertex count, instance count are of this type. 
    pub fn count(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Count)
    }

    /// Trace the flow head to generate a flow graph.
    pub fn graph(&self, head: &FlowHead) -> FlowGraph {
        FlowGraph::new(head, &self.flows)
    }
}

pub struct DeviceProcInner {
    dev: Arc<DeviceInner>,
    graph: FlowGraph,
    name_map: HashMap<String, usize>,
    ty_map: HashMap<usize, VariableType>,
}
impl_ptr_wrapper!(DeviceProc -> DeviceProcInner);
pub struct DeviceProc(Arc<DeviceProcInner>);
impl DeviceProc {
    // TODO: Pass in dev to ensure all pipelines are constructed on the same
    // device.
    pub fn new<'a, F>(dev: &Device, graph_fn: F) -> DeviceProc
        where F: FnOnce(&mut SymbolSource) -> FlowGraph
    {
        let dev = dev.0.clone();
        let mut sym_src = SymbolSource::new();
        let graph = graph_fn(&mut sym_src);
        let SymbolSource { syms, name_map, .. } = sym_src;
        let ty_map = name_map.values()
            .map(|token| (*token, syms[*token]))
            .collect::<HashMap<_, _>>();

        let inner = DeviceProcInner { dev, graph, name_map, ty_map };
        DeviceProc(Arc::new(inner))
    }
}

#[derive(Clone)]
pub enum Variable<'a> {
    HostMemory(&'a [u8]),
    Buffer(Buffer),
    Image(Image),
    Sampler(Sampler),
    SampledImage(Image, Sampler),
    Count(u32),
}
impl<'a> Variable<'a> {
    pub fn ty(&self) -> VariableType {
        match self {
            Variable::HostMemory(_) => VariableType::HostMemory,
            Variable::Buffer(_) => VariableType::Buffer,
            Variable::Image(_) => VariableType::Image,
            Variable::Sampler(_) => VariableType::Sampler,
            Variable::SampledImage(_, _) => VariableType::SampledImage,
            Variable::Count(_) => VariableType::Count,
        }
    }
    pub fn is_instance_of(&self, ty: VariableType) -> bool {
        self.ty() == ty
    }
    pub fn to_host_mem(&self) -> Option<&'a [u8]> {
        if let Variable::HostMemory(x) = self { Some(x) } else { None }
    }
    pub fn to_buf(&self) -> Option<&Buffer> {
        if let Variable::Buffer(x) = self { Some(x) } else { None }
    }
    pub fn to_img(&self) -> Option<&Image> {
        if let Variable::Image(x) = self { Some(x) } else { None }
    }
    pub fn to_sampler(&self) -> Option<&Sampler> {
        if let Variable::Sampler(x) = self { Some(x) } else { None }
    }
    pub fn to_sampled_img(&self) -> Option<(&Image, &Sampler)> {
        if let Variable::SampledImage(x, y) = self {
            Some((x, y))
        } else { None }
    }

    pub fn to_count(&self) -> Option<u32> {
        if let Variable::Count(x) = self { Some(*x) } else { None }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RwState {
    Write, Read
}
impl Default for RwState {
    fn default() -> RwState {
        RwState::Write
    }
}
struct VariableState<'a> {
    var: &'a Variable<'a>,
    stage: vk::PipelineStageFlags,
    access: vk::AccessFlags,
    qfam_idx: u32,
    rw: RwState,
}
impl<'a> VariableState<'a> {
    fn new(var: &'a Variable<'a>) -> VariableState<'a> {
        VariableState {
            var,
            stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            access: vk::AccessFlags::HOST_WRITE,
            qfam_idx: !0,
            rw: RwState::Read,
        }
    }
}

struct Recorder<'a> {
    dev: &'a DeviceInner,
    cmdbuf: vk::CommandBuffer,
    state_dict: &'a HashMap<VariableToken, RefCell<VariableState<'a>>>,

    desc_pools: &'a mut Vec<vk::DescriptorPool>,
    framebufs: &'a mut Vec<vk::Framebuffer>,

    push_const: Option<&'a [u8]>,
    // set -> bind -> var
    desc_binds: HashMap<u32, HashMap<u32, VariableToken>>,
    // attr bp -> var
    vert_bufs: HashMap<u32, VariableToken>,
    idx_buf: Option<VariableToken>,
    // attm idx -> var
    attms: HashMap<u32, VariableToken>,
}
impl<'a> Recorder<'a> {
    fn create_desc_pool(
        dev: &ash::Device,
        shader_arr: &ShaderArrayInner,
    ) -> Result<vk::DescriptorPool> {
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(shader_arr.desc_set_layouts.len() as u32)
            .pool_sizes(&shader_arr.desc_pool_sizes)
            .build();
        let desc_pool = unsafe {
            dev.create_descriptor_pool(&create_info, None)?
        };
        Ok(desc_pool)
    }
    fn alloc_desc_set(
        dev: &ash::Device,
        desc_pool: vk::DescriptorPool,
        shader_arr: &ShaderArrayInner,
        set: u32,
    ) -> Result<vk::DescriptorSet> {
        let desc_set_layout = [shader_arr.desc_set_layouts[&set]];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&desc_set_layout)
            .build();
        let desc_set = unsafe {
            dev.allocate_descriptor_sets(&alloc_info)?
        };
        Ok(desc_set[0])
    }
    fn create_framebuf(
        dev: &ash::Device,
        pass: &RenderPassInner,
        attms: Vec<vk::ImageView>,
    ) -> Result<vk::Framebuffer> {
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(pass.pass)
            .attachments(&attms)
            .width(pass.framebuf_extent.0)
            .height(pass.framebuf_extent.1)
            .layers(1) // TODO: Support multilayer rendering in the future.
            .build();
        let framebuf = unsafe { dev.create_framebuffer(&create_info, None)? };
        Ok(framebuf)
    }

    fn new(
        dev: &'a DeviceInner,
        state_dict: &'a HashMap<VariableToken, RefCell<VariableState<'a>>>,
        desc_pools: &'a mut Vec<vk::DescriptorPool>,
        framebufs: &'a mut Vec<vk::Framebuffer>,
    ) -> Recorder<'a> {
        Recorder {
            dev,
            cmdbuf: vk::CommandBuffer::null(),
            state_dict,
            desc_pools,
            framebufs,
            push_const: None,
            desc_binds: HashMap::new(),
            vert_bufs: HashMap::new(),
            idx_buf: None,
            attms: HashMap::new(),
        }
    }

    // Get the variable referred by `token` and keep track of the state of the
    // dynamic variable.
    fn stage_var(
        &self,
        token: VariableToken,
        stage: vk::PipelineStageFlags,
        access: vk::AccessFlags,
        qfam_idx: u32,
        rw: RwState,
    ) -> &Variable<'a> {
        fn make_buf_bar<'a>(
            buf: &BufferInner,
            state: &VariableState<'a>,
            access: vk::AccessFlags,
            qfam_idx: u32,
        ) -> vk::BufferMemoryBarrier {
            vk::BufferMemoryBarrier::builder()
                .src_access_mask(state.access)
                .dst_access_mask(access)
                .src_queue_family_index(state.qfam_idx)
                .dst_queue_family_index(qfam_idx)
                .buffer(buf.buf)
                .offset(0)
                .size(buf.cfg.size as u64)
                .build()
        }
        fn make_img_bar<'a>(
            img: &ImageInner,
            state: &VariableState<'a>,
            access: vk::AccessFlags,
            qfam_idx: u32,
        ) -> vk::ImageMemoryBarrier {
            let subrsc_rng = vk::ImageSubresourceRange {
                aspect_mask: img.aspect,
                base_mip_level: 0,
                level_count: img.cfg.nmip,
                base_array_layer: 0,
                layer_count: img.cfg.nlayer,
            };
            let old_layout = img.layout.get();
            let new_layout = vk::ImageLayout::GENERAL;
            img.layout.set(new_layout);
            vk::ImageMemoryBarrier::builder()
                .src_access_mask(state.access)
                .dst_access_mask(access)
                .old_layout(old_layout)
                // TODO: (penguinliong) Adapt to different accesses, or, should
                // we?
                .new_layout(new_layout)
                .src_queue_family_index(state.qfam_idx)
                .dst_queue_family_index(qfam_idx)
                .image(img.img)
                .subresource_range(subrsc_rng)
                .build()
        }
        let mut state = self.state_dict[&token].borrow_mut();
        // Don't barrier if the previous and the next RW state are both `Read`.
        if (rw == RwState::Write) | (state.rw == RwState::Write) {
            let bars = match state.var {
                Variable::Buffer(buf) => {
                    (
                        vec![make_buf_bar(&buf.0, &state, access, qfam_idx)],
                        vec![],
                    )
                },
                Variable::Image(img) => {
                    (
                        vec![],
                        vec![make_img_bar(&img.0, &state, access, qfam_idx)],
                    )
                },
                Variable::SampledImage(img, _) => {
                    (
                        vec![],
                        vec![make_img_bar(&img.0, &state, access, qfam_idx)],
                    )
                },
                // Don't care readonly types.
                _ => return &state.var,
            };
            let (buf_bar, img_bar) = bars;
            unsafe {
                self.dev.dev.cmd_pipeline_barrier(
                    self.cmdbuf,
                    state.stage,
                    stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &buf_bar,
                    &img_bar,
                );
            }
        }
        state.stage = stage;
        state.access = access;
        state.qfam_idx = qfam_idx;
        state.rw = rw;
        &state.var
    }
    fn stage_var_host_const(
        &mut self,
        token: VariableToken,
    ) -> &Variable<'a> {
        self.stage_var(token, Default::default(), Default::default(), !0,
            RwState::Read)
    }

    fn bind_var(&mut self, bp: BindPoint, token: VariableToken) {
        match bp {
            BindPoint::Descriptor(set, bind) => {
                *self.desc_binds.entry(set).or_default()
                    .entry(bind).or_default() = token;
            },
            BindPoint::VertexInput(i) => {
                *self.vert_bufs.entry(i).or_default() = token;
            },
            BindPoint::Index => {
                self.idx_buf = Some(token);
            },
            BindPoint::Attachment(i) => {
                *self.attms.entry(i).or_default() = token;
            },
        }
    }
    fn unbind_var(&mut self, bp: BindPoint) {
        match bp {
            BindPoint::Descriptor(set, bind) => {
                self.desc_binds.entry(set)
                    .and_modify(|x| { x.remove(&bind); });
            },
            BindPoint::VertexInput(i) => {
                self.vert_bufs.remove(&i);
            },
            BindPoint::Index => {
                self.idx_buf = None;
            },
            BindPoint::Attachment(i) => {
                self.attms.remove(&i);
            },
        }
    }

    #[inline]
    fn bind_pipe(&self, pipe: &PipelineInner) {
        unsafe {
            self.dev.dev.cmd_bind_pipeline(self.cmdbuf, pipe.pipe_bp, pipe.pipe);
        }
    }
    fn flush_push_const(&self, pipe: &PipelineInner) -> Result<()> {
        if let Some(push_const) = self.push_const {
            unsafe {
                self.dev.dev.cmd_push_constants(
                    self.cmdbuf,
                    pipe.pipe_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    push_const,
                );
            }
        }
        Ok(())
    }
    #[inline]
    fn clear_push_const(&mut self) {
        self.push_const = None;
    }
    fn flush_desc_binds(
        &mut self,
        pipe: &PipelineInner,
        qfam_idx: u32,
    ) -> Result<()> {
        let dev = &self.dev.dev;
        let shader_arr = &*pipe.shader_arr;
        let desc_pool = Self::create_desc_pool(dev, shader_arr)?;
        // TODO: (penguinliong) Remember to destroy this.
        // Note that allocated desc sets will be freed on destroy.
        self.desc_pools.push(desc_pool);
        let qfam_idx = qfam_idx;
        // Bind descriptor sets.
        for (set, bind_var) in self.desc_binds.iter() {
            let desc_set = {
                Self::alloc_desc_set(dev, desc_pool, shader_arr, *set)?
            };
            // DO NOT TRIGGER ANY REALLOC OR THERE WILL BE DANGLING POINTERS!!
            let n = bind_var.len();
            let mut buf_infos = Vec::with_capacity(n);
            let mut img_infos = Vec::with_capacity(n);
            let mut write_desc_sets = Vec::with_capacity(n);
            for (bind, token) in bind_var {
                let write_desc_set = {
                    let write_base = |ty| {
                        vk::WriteDescriptorSet::builder()
                            .dst_set(desc_set)
                            .dst_binding(*bind)
                            .dst_array_element(0)
                            .descriptor_type(ty)
                    };
                    let mut write_buf = |ty, buf: Buffer| {
                        let buf_info = vk::DescriptorBufferInfo {
                            buffer: buf.buf,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        };
                        let i = buf_infos.len();
                        buf_infos.push(buf_info);
                        write_base(ty)
                            .buffer_info(&buf_infos[i..i + 1])
                            .build()
                    };
                    let mut write_img = 
                        |ty, img: Option<Image>, sampler: Option<Sampler>| {
                            let sampler = sampler
                                .map(|x| x.sampler)
                                .unwrap_or(vk::Sampler::null());
                            let (image_view, image_layout) = img
                                .map(|x| (x.img_view, x.layout.get()))
                                .unwrap_or((
                                    vk::ImageView::null(),
                                    vk::ImageLayout::UNDEFINED,
                                ));
                            let img_info = vk::DescriptorImageInfo {
                                sampler,
                                image_view,
                                image_layout,
                            };
                            let i = img_infos.len();
                            img_infos.push(img_info);
                            write_base(ty)
                                .image_info(&img_infos[i..i + 1])
                                .build()
                        };

                    let desc_bind = spirq::DescriptorBinding::new(*set, *bind);
                    let desc_ty = pipe.shader_arr.manifest.get_desc(desc_bind)
                        .ok_or(Error::InvalidOperation)?;
                    let desc_ty = spirq_desc_ty2vk_desc_ty(desc_ty);
                    match desc_ty {
                        _ => unreachable!(),
                        vk::DescriptorType::UNIFORM_BUFFER => {
                            let buf = self.stage_var(
                                    *token,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::UNIFORM_READ,
                                    qfam_idx,
                                    RwState::Read,
                                )
                                .to_buf()
                                .cloned()
                                .ok_or(Error::InvalidOperation)?;
                            write_buf(desc_ty, buf)
                        },
                        vk::DescriptorType::STORAGE_BUFFER => {
                            let buf = self.stage_var(
                                    *token,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::SHADER_READ |
                                    vk::AccessFlags::SHADER_WRITE,
                                    qfam_idx,
                                    // TODO: (penguinliong) Allow SPIR-Q to
                                    //       extract `readonly` decorations. 
                                    RwState::Write,
                                )
                                .to_buf()
                                .cloned()
                                .ok_or(Error::InvalidOperation)?;
                            write_buf(desc_ty, buf)
                        },
                        vk::DescriptorType::SAMPLED_IMAGE => {
                            let img = self.stage_var(
                                    *token,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::SHADER_READ,
                                    qfam_idx,
                                    RwState::Read,
                                )
                                .to_img()
                                .cloned()
                                .ok_or(Error::InvalidOperation)?;
                            write_img(desc_ty, Some(img), None)
                        },
                        vk::DescriptorType::STORAGE_IMAGE => {
                            let img = self.stage_var(
                                    *token,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::SHADER_READ |
                                    vk::AccessFlags::SHADER_WRITE,
                                    qfam_idx,
                                    // TODO: (penguinliong) Allow SPIR-Q to
                                    //       extract `readonly` decorations. 
                                    RwState::Write,
                                )
                                .to_img()
                                .cloned()
                                .ok_or(Error::InvalidOperation)?;
                            write_img(desc_ty, Some(img), None)
                        },
                        vk::DescriptorType::SAMPLER => {
                            let sampler = self.stage_var(
                                    *token,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::SHADER_READ,
                                    qfam_idx,
                                    RwState::Read,
                                )
                                .to_sampler()
                                .cloned()
                                .ok_or(Error::InvalidOperation)?;
                            write_img(desc_ty, None, Some(sampler))
                        }
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                            let (img, sampler) = self.stage_var(
                                    *token,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::SHADER_READ,
                                    qfam_idx,
                                    RwState::Read,
                                )
                                .to_sampled_img()
                                .ok_or(Error::InvalidOperation)?;
                            write_img(desc_ty, Some(img.clone()),
                                Some(sampler.clone()))
                        },
                        // TODO: (penguinliong) Check out where does input
                        // attachment go.
                    }
                };
                write_desc_sets.push(write_desc_set);
            }
            unsafe {
                dev.update_descriptor_sets(&write_desc_sets, &[]);
                dev.cmd_bind_descriptor_sets(
                    self.cmdbuf,
                    pipe.pipe_bp,
                    pipe.pipe_layout,
                    *set,
                    &[desc_set],
                    &[],
                );
            }
            // We can safely discard the descriptor set here, because it will be
            // freed when the descriptor pool is destroyed which we will do in
            // `drop()`.
        }
        Ok(())
    }
    #[inline]
    fn clear_desc_binds(&mut self) {
        self.desc_binds.clear();
    }
    fn flush_vert_bufs(&self, qfam_idx: u32) -> Result<()> {
        for (vert_bind, token) in self.vert_bufs.iter() {
            let vert_buf = self.stage_var(
                    *token,
                    vk::PipelineStageFlags::VERTEX_INPUT,
                    vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                    qfam_idx,
                    RwState::Read,
                )
                .to_buf()
                .ok_or(Error::InvalidOperation)?;
            unsafe {
                self.dev.dev.cmd_bind_vertex_buffers(
                    self.cmdbuf,
                    *vert_bind,
                    &[vert_buf.buf],
                    &[0],
                );
            }
        }
        Ok(())
    }
    #[inline]
    fn clear_vert_bufs(&mut self) {
        self.vert_bufs.clear();
    }
    fn flush_idx_buf(&mut self, qfam_idx: u32) -> Result<()> {
        if let Some(token) = self.idx_buf.take() {
            let idx_buf = self.stage_var(
                    token,
                    vk::PipelineStageFlags::VERTEX_INPUT,
                    vk::AccessFlags::INDEX_READ,
                    qfam_idx,
                    RwState::Read,
                )
                .to_buf()
                .ok_or(Error::InvalidOperation)?;
            unsafe {
                self.dev.dev.cmd_bind_index_buffer(
                    self.cmdbuf, idx_buf.buf, 0, vk::IndexType::UINT16
                );
            }
        }
        Ok(())
    }
    #[inline]
    fn clear_idx_buf(&mut self) {
        self.idx_buf = None;
    }
    fn flush_attms(
        &mut self,
        qfam_idx: u32,
        pass: &RenderPassInner,
    ) -> Result<vk::Framebuffer> {
        let dev = &self.dev.dev;
        let mut attms = Vec::with_capacity(self.attms.len());
        for token in self.attms.values() {
            let attm = self.stage_var(
                    *token,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE |
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    qfam_idx,
                    RwState::Write,
                )
                .to_img()
                .ok_or(Error::InvalidOperation)?;
            attms.push(attm.img_view);
        }

        Self::create_framebuf(dev, pass, attms)
    }
    #[inline]
    fn clear_attms(&mut self) {
        self.attms.clear();
    }

    fn record_transfer(&mut self, args: &TransferEventArgs) -> Result<()> {
        fn buf_copy(src: &BufferInner, dst: &BufferInner) -> vk::BufferCopy {
            vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src.cfg.size as u64,
            }
        }
        fn img_copy(src: &ImageInner, dst: &ImageInner) -> vk::ImageCopy {
            let src_subrsc = vk::ImageSubresourceLayers {
                aspect_mask: src.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: src.cfg.nlayer,
            };
            let dst_subrsc = vk::ImageSubresourceLayers {
                aspect_mask: dst.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: dst.cfg.nlayer,
            };
            let extent = vk::Extent3D {
                width: src.cfg.width,
                height: src.cfg.height,
                depth: src.cfg.depth,
            };
            vk::ImageCopy {
                src_subresource: src_subrsc,
                src_offset: vk::Offset3D::default(),
                dst_subresource: dst_subrsc,
                dst_offset: vk::Offset3D::default(),
                extent,
            }
        }
        fn buf_img_copy(
            buf: &BufferInner,
            img: &ImageInner,
        ) -> vk::BufferImageCopy {
            let subrsc = vk::ImageSubresourceLayers {
                aspect_mask: img.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: img.cfg.nlayer,
            };
            let extent = vk::Extent3D {
                width: img.cfg.width,
                height: img.cfg.height,
                depth: img.cfg.depth,
            };
            vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: img.cfg.width,
                buffer_image_height: img.cfg.height,
                image_subresource: subrsc,
                image_offset: vk::Offset3D::default(),
                image_extent: extent,
            }
        }
        let qfam_idx = self.dev.qmap
            .get(&QueueInterface::Transfer)
            .ok_or(Error::InvalidOperation)?
            .qloc
            .qfam_idx;
        let dev = &self.dev.dev;
        let src_var = self.stage_var(
                args.src_var_idx,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
                qfam_idx,
                RwState::Read,
            );
        let dst_var = self.stage_var(
            args.dst_var_idx,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            qfam_idx,
            RwState::Write, 
        );
        match (src_var, dst_var) {
            (Variable::Buffer(src), Variable::Buffer(dst)) => {
                unsafe {
                    dev.cmd_copy_buffer(
                        self.cmdbuf,
                        src.buf,
                        dst.buf,
                        &[buf_copy(&*src, &*dst)],
                    );
                }
            },
            (Variable::Image(src), Variable::Image(dst)) => {
                unsafe {
                    dev.cmd_copy_image(
                        self.cmdbuf,
                        src.img,
                        vk::ImageLayout::GENERAL,
                        dst.img,
                        vk::ImageLayout::GENERAL,
                        &[img_copy(&*src, &*dst)],
                    );
                }
            },
            (Variable::Buffer(src), Variable::Image(dst)) => {
                unsafe {
                    dev.cmd_copy_buffer_to_image(
                        self.cmdbuf,
                        src.buf,
                        dst.img,
                        vk::ImageLayout::GENERAL,
                        &[buf_img_copy(&*src, &*dst)],
                    );
                }
            },
            (Variable::Image(src), Variable::Buffer(dst)) => {
                unsafe {
                    dev.cmd_copy_image_to_buffer(
                        self.cmdbuf,
                        src.img,
                        vk::ImageLayout::GENERAL,
                        dst.buf,
                        &[buf_img_copy(&*dst, &*src)],
                    );
                }
            },
            _ => return Err(Error::InvalidOperation),
        }
        Ok(())
    }
    fn record_push_const(
        &mut self,
        args: &PushConstantEventArgs,
    ) -> Result<()> {
        let host_mem = self.stage_var_host_const(args.push_const_var_idx)
            .to_host_mem()
            .ok_or(Error::InvalidOperation)?;
        self.push_const = Some(host_mem);
        Ok(())
    }
    fn record_bind(&mut self, args: &BindEventArgs) {
        if let Some(token) = args.var_idx.as_ref() {
            self.bind_var(args.bp, *token);
        } else {
            self.unbind_var(args.bp);
        }
    }
    fn record_dispatch(
        &mut self,
        args: &DispatchEventArgs,
    ) -> Result<()> {
        let qfam_idx = self.dev.qmap.get(&QueueInterface::Compute)
            .ok_or(Error::InvalidOperation)?
            .qloc
            .qfam_idx;

        let x = self.stage_var_host_const(args.nworkgrp_x_var_idx)
            .to_count()
            .ok_or(Error::InvalidOperation)?;
        let y = self.stage_var_host_const(args.nworkgrp_y_var_idx)
            .to_count()
            .ok_or(Error::InvalidOperation)?;
        let z = self.stage_var_host_const(args.nworkgrp_z_var_idx)
            .to_count()
            .ok_or(Error::InvalidOperation)?;

        let dev = &self.dev.dev;
        for comp_pipe in args.task.pipes.iter() {
            self.flush_desc_binds(&comp_pipe, qfam_idx)?;
            self.flush_push_const(&comp_pipe)?;
            self.bind_pipe(&comp_pipe);
            unsafe { dev.cmd_dispatch(self.cmdbuf, x, y, z); }
        }

        self.clear_push_const();
        self.clear_desc_binds();
        Ok(())
    }
    fn record_draw(&mut self, args: &DrawEventArgs) -> Result<()> {
        let qfam_idx = self.dev.qmap.get(&QueueInterface::Graphics)
            .ok_or(Error::InvalidOperation)?
            .qloc
            .qfam_idx;

        let no_idx = self.idx_buf.is_none();
        let framebuf = self.flush_attms(qfam_idx, &*args.pass)?;
        self.framebufs.push(framebuf);
        self.flush_vert_bufs(qfam_idx)?;
        self.flush_idx_buf(qfam_idx)?;

        let nvert = self.stage_var_host_const(args.nvert_var_idx)
            .to_count()
            .ok_or(Error::InvalidOperation)?;
        let ninst = self.stage_var_host_const(args.ninst_var_idx)
            .to_count()
            .ok_or(Error::InvalidOperation)?;

        let dev = &self.dev.dev;
        let mut first = true;
        for pipe in args.pass.pipes.iter() {
            self.flush_push_const(&pipe);
            self.flush_desc_binds(&pipe, qfam_idx)?;
            if first {
                let pass = &*args.pass;
                let render_area = vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: pass.framebuf_extent.0,
                        height: pass.framebuf_extent.1,
                    },
                };
                let begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(pass.pass)
                    .framebuffer(framebuf)
                    .render_area(render_area)
                    .build();
                // TODO: (penguinliong) Support attachment clear values.
                // TODO: (penguinliong) Use secondary command buffer if
                //       possible.
                unsafe {
                    dev.cmd_begin_render_pass(
                        self.cmdbuf, &begin_info, vk::SubpassContents::INLINE
                    );
                }
                first = false;
            } else {
                unsafe {
                    dev.cmd_next_subpass(
                        self.cmdbuf, vk::SubpassContents::INLINE
                    );
                }
            }

            self.flush_push_const(&pipe)?;
            self.bind_pipe(&pipe);
            if no_idx {
                unsafe {
                    dev.cmd_draw(self.cmdbuf, nvert, ninst, 0, 0);
                }
            } else {
                unsafe {
                    dev.cmd_draw_indexed(self.cmdbuf, nvert, ninst, 0, 0, 0);
                }
            }
        }

        self.clear_push_const();
        self.clear_desc_binds();
        self.clear_idx_buf();
        self.clear_vert_bufs();
        self.clear_attms();
        Ok(())
    }
    fn record_chunk(
        &mut self,
        dev: &ash::Device,
        cmdbuf: vk::CommandBuffer,
        chunk: &Chunk,
    ) -> Result<()> {
        self.cmdbuf = cmdbuf;
        let qi = chunk.qi.unwrap();
        for event in chunk.events.iter() {
            match event {
                Event::Transfer(args) => self.record_transfer(args)?,
                Event::PushConstant(args) => self.record_push_const(args)?,
                Event::Bind(args) => self.record_bind(args),
                Event::Dispatch(args) => self.record_dispatch(args)?,
                Event::Draw(args) => self.record_draw(args)?,
            }
        }
        Ok(())
    }
}

struct TransactionSubmitDetail {
    queue: vk::Queue,
    cmdbuf: vk::CommandBuffer,
    wait_semas: Vec<vk::Semaphore>,
    // Every submit will signal a semaphore, even if there is nothing waiting
    // for it.
    signal_sema: vk::Semaphore,
}
pub struct TransactionInner {
    devproc: Arc<DeviceProcInner>,
    cmdpools: HashMap<vk::Queue, vk::CommandPool>,
    desc_pools: Vec<vk::DescriptorPool>,
    framebufs: Vec<vk::Framebuffer>,
    submit_details: Vec<TransactionSubmitDetail>,
    fence: vk::Fence,
}
impl Drop for TransactionInner {
    fn drop(&mut self) {
        let dev = &self.devproc.dev.dev;
        for cmdpool in self.cmdpools.values() {
            unsafe { dev.destroy_command_pool(*cmdpool, None); }
        }
        for desc_pool in self.desc_pools.iter() {
            unsafe { dev.destroy_descriptor_pool(*desc_pool, None); }
        }
        for framebuf in self.framebufs.iter() {
            unsafe { dev.destroy_framebuffer(*framebuf, None); }
        }
    }
}
pub struct Transaction(Box<TransactionInner>);
impl Transaction {
    fn create_cmdpool(
        dev: &DeviceInner,
        qfam_idx: u32,
    ) -> Result<vk::CommandPool> {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(qfam_idx)
            .build();
        let cmdpool = unsafe {
            dev.dev.create_command_pool(&create_info, None)?
        };
        Ok(cmdpool)
    }
    fn ensure_cmdpool(
        dev: &DeviceInner,
        qfam_idx: u32,
        queue: vk::Queue,
        cmdpools: &mut HashMap<vk::Queue, vk::CommandPool>,
    ) -> Result<vk::CommandPool> {
        use std::collections::hash_map::Entry::{Vacant, Occupied};
        let rv = match cmdpools.entry(queue) {
            Vacant(entry) => {
                Self::create_cmdpool(dev, qfam_idx)
                    .map(|cmdpool| {
                        entry.insert(cmdpool);
                        cmdpool
                    })
            },
            Occupied(entry) => Ok(*entry.get()),
        };
        if rv.is_err() {
            for cmdpool in cmdpools.values() {
                // Make sure previously allocated cmdpools are all destroyed.
                unsafe { dev.dev.destroy_command_pool(*cmdpool, None) };
            }
        }
        rv
    }
    fn alloc_cmdbuf(
        dev: &DeviceInner,
        cmdpool: vk::CommandPool,
    ) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(cmdpool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();
        let cmdbuf = unsafe { dev.dev.allocate_command_buffers(&alloc_info)? };
        Ok(cmdbuf[0])
    }
    fn create_sema(dev: &DeviceInner) -> Result<vk::Semaphore> {
        let create_info = vk::SemaphoreCreateInfo::builder()
            .build();
        let sema = unsafe { dev.dev.create_semaphore(&create_info, None)? };
        Ok(sema)
    }
    fn create_fence(dev: &DeviceInner) -> Result<vk::Fence> {
        let create_info = vk::FenceCreateInfo::builder()
            .build();
        let fence = unsafe { dev.dev.create_fence(&create_info, None)? };
        Ok(fence)
    }
    fn create_submit_details(
        dev: &DeviceInner,
        graph: &FlowGraph,
        mut cmdpools: &mut HashMap<vk::Queue, vk::CommandPool>,
    ) -> Result<Vec<TransactionSubmitDetail>> {
        let mut submit_details = {
            Vec::<TransactionSubmitDetail>::with_capacity(graph.chunks.len())
        };
        let ilast = graph.chunks.len() - 1;
        for (i, chunk) in graph.chunks.iter().enumerate() {
            let queue_detail = chunk.qi
                // Unbound chunks are not allowed. It's also not allowed to do
                // anything on a unsupported queue interface.
                .and_then(|qi| dev.qmap.get(&qi))
                .ok_or(Error::InvalidOperation)?;
            let queue = queue_detail.queue;
            let qfam_idx = queue_detail.qloc.qfam_idx;

            // No need to optimize recording using the same command buffer
            // when consequential chunks use the same queue. If any two
            // consequential chunks have the same destination queue, while
            // are not fused in one, it must be forcefully divided by the
            // user, when they design the flow graph.
            let cmdpool = {
                Self::ensure_cmdpool(dev, qfam_idx, queue, &mut cmdpools)?
            };
            let cmdbuf = Self::alloc_cmdbuf(dev, cmdpool)?;
            let wait_semas = graph.rev_dep_map[i].iter()
                .map(|i| submit_details[*i].signal_sema)
                .collect();
            let signal_sema = Self::create_sema(dev)?;
            let submit_detail = TransactionSubmitDetail {
                queue, cmdbuf, wait_semas, signal_sema
            };
            submit_details.push(submit_detail);
        }
        Ok(submit_details)
    }
    pub fn new(devproc: &DeviceProc) -> Result<Transaction> {
        let devproc = devproc.0.clone();
        let dev = &*devproc.dev;
        let mut cmdpools = HashMap::new();
        let desc_pools = Vec::new();
        let framebufs = Vec::new();
        let submit_details = {
            Self::create_submit_details(dev, &devproc.graph, &mut cmdpools)?
        };
        let fence = Self::create_fence(dev)?;
        let inner = TransactionInner {
            devproc, cmdpools, desc_pools, framebufs, submit_details, fence
        };
        Ok(Transaction(Box::new(inner)))
    }
    pub fn arm<'a>(
        self,
        var_dict: &HashMap<String, Variable<'a>>,
    ) -> Result<ArmedTransaction<'a>> {
        let mut transact = self.0;
        let devproc = &*transact.devproc;
        let dev = &*devproc.dev;
        let name_map = &devproc.name_map;
        let mut state_dict = HashMap::new();
        for (name, token) in name_map.iter() {
            let var = var_dict.get(name)
                .ok_or(Error::InvalidOperation)?;
            state_dict.insert(*token, RefCell::new(VariableState::new(var)));
        }

        let mut rec = Recorder::new(dev, &state_dict,
            &mut transact.desc_pools, &mut transact.framebufs);

        for (i, chunk) in devproc.graph.chunks.iter().enumerate() {
            let submit_detail = &transact.submit_details[i];
            rec.record_chunk(&dev.dev, submit_detail.cmdbuf, &chunk)?;
        }
        Ok(ArmedTransaction(transact, PhantomData))
    }
}
pub struct ArmedTransaction<'a>(
    Box<TransactionInner>,
    PhantomData<&'a Variable<'a>>,
);
impl<'a> ArmedTransaction<'a> {
    fn submit_impl(transact: &TransactionInner) -> Result<()> {
        let dev = &transact.devproc.dev.dev;
        let ilast = transact.submit_details.len() - 1;
        for (i, submit_detail) in transact.submit_details.iter().enumerate() {
            let queue = submit_detail.queue;
            let signal_sema = [submit_detail.signal_sema];
            let cmdbuf = [submit_detail.cmdbuf];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&submit_detail.wait_semas)
                .signal_semaphores(&signal_sema)
                .command_buffers(&cmdbuf)
                .build();
            let fence = if i == ilast {
                transact.fence
            } else {
                vk::Fence::null()
            };
            unsafe { dev.queue_submit(queue, &[submit_info], fence)? };
        }
        Ok(())
    }
    fn present_impl(
        transact: &TransactionInner,
        swapchain_img: &SwapchainImage,
    ) -> Result<()> {
        let dev = &transact.devproc.dev;
        let khr_swapchain = &dev.cap_detail.dev_exts.khr_swapchain.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let present_detail = dev.present_detail.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let wait_sema = {
            if let Some(submit_detail) = transact.submit_details.last() {
                [submit_detail.signal_sema]
            } else {
                [vk::Semaphore::null()]
            }
        };
        let swapchain = [present_detail.swapchain];
        let idx = [swapchain_img.img_idx];
        let queue = dev.qmap[&QueueInterface::Present].queue;
        let present_info = {
            let mut x = vk::PresentInfoKHR::builder();
            if wait_sema[0] == vk::Semaphore::null() {
                x = x.wait_semaphores(&wait_sema);
            }
            x.swapchains(&swapchain)
                .image_indices(&idx)
                .build()
        };
        unsafe { khr_swapchain.queue_present(queue, &present_info)? };
        Ok(())
    }
    /// Submit the command buffers to the Vulkan implementation for execution.
    pub fn submit<'b: 'a>(self) -> Result<PendingTransaction<'b>> {
        let transact = self.0;
        {
            let transact = &*transact;
            Self::submit_impl(transact)?;
        }
        Ok(PendingTransaction(transact, PhantomData))
    }
    /// Submit the command buffers to the Vulkan implementation for execution
    /// and present the latest acquired swapchain image.
    pub fn submit_present<'b: 'a>(
        self,
        swapchain_img: &SwapchainImage,
    ) -> Result<PendingTransaction<'b>> {
        let transact = self.0;
        {
            let transact = &*transact;
            Self::submit_impl(transact)?;
            Self::present_impl(transact, swapchain_img)?;
        }
        Ok(PendingTransaction(transact, PhantomData))
    }
}
pub struct PendingTransaction<'a>(
    Box<TransactionInner>,
    PhantomData<&'a Variable<'a>>,
);
impl<'a> PendingTransaction<'a> {
    pub fn wait(
        &self,
        timeout: u64,
    ) -> std::result::Result<(), ()> {
        let transact = &*self.0;
        let dev = &transact.devproc.dev.dev;
        // It doesn't matter whether to wait for all fences tho, but it should
        // be noticed if we are accepting multiple fences in the future.
        let res = unsafe {
            dev.wait_for_fences(&[transact.fence], true, timeout)
        };
        match res {
            Err(vk::Result::TIMEOUT) => Err(()),
            Ok(_) => Ok(()),
            Err(_) => unreachable!(),
        }
    }
    pub fn reset(self) -> Result<Transaction> {
        let mut transact = self.0;
        let dev = &transact.devproc.dev.dev;
        // Reset everything. (Also note that semaphores don't need to be reset.)
        unsafe { dev.reset_fences(&[transact.fence]) }?;
        for cmdpool in transact.cmdpools.values() {
            unsafe { dev.reset_command_pool(*cmdpool, Default::default()) }?;
        }
        for desc_pool in transact.desc_pools.iter() {
            unsafe { dev.destroy_descriptor_pool(*desc_pool, None); }
        }
        transact.desc_pools.clear();
        for framebuf in transact.framebufs.iter() {
            unsafe { dev.destroy_framebuffer(*framebuf, None); }
        }
        transact.framebufs.clear();
        Ok(Transaction(transact))
    }
}

// TODO: (penguinliong): Clean up for all creation failures.
// TODO: (penguinliong): Defer image layout change to when it's actually
//       changed, i.e., after the wait.
