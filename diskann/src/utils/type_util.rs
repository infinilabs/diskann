use std::any::Any;

pub fn is_f16<T: Any>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() // Placeholder for f16
}

pub fn is_f32<T: Any>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
}

pub fn is_f64<T: Any>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
}

pub fn is_f128<T: Any>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() // Placeholder for f128
}

pub fn is_floating_point<T: Any>() -> bool {
    is_f16::<T>() || is_f32::<T>() || is_f64::<T>() || is_f128::<T>()
}
