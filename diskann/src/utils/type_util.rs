use std::any::{Any, TypeId};
pub fn is_f16<T: Any>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f16>()
}

pub fn is_f32<T: Any>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

pub fn is_f64<T: Any>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

pub fn is_f128<T: Any>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f128>()
}

pub fn is_floating_point<T: Any>() -> bool {
    is_f16() || is_f32() || is_f64() || is_f128()
}
