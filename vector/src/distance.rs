/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::l2_float_distance::{
    distance_cosine_vector_f32, distance_l2_vector_f16, distance_l2_vector_f32,
};
use crate::{Half, Metric};

/// Distance contract for full-precision vertex
pub trait FullPrecisionDistance<T, const N: usize> {
    /// Get the distance between vertex a and vertex b
    fn distance_compare(a: &[T; N], b: &[T; N], vec_type: Metric) -> f32;
}

// reason = "Not supported Metric type Metric::Cosine"
#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<f32, N> for [f32; N] {
    /// Calculate distance between two f32 Vertex
    #[inline(always)]
    fn distance_compare(a: &[f32; N], b: &[f32; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => distance_l2_vector_f32::<N>(a, b),
            Metric::Cosine => distance_cosine_vector_f32::<N>(a, b),
            //_ => panic!("Not supported Metric type {:?}", metric),
        }
    }
}

// reason = "Not supported Metric type Metric::Cosine"
#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<Half, N> for [Half; N] {
    fn distance_compare(a: &[Half; N], b: &[Half; N], metric: Metric) -> f32 {
        match metric {
            Metric::L2 => distance_l2_vector_f16::<N>(a, b),
            _ => panic!("Not supported Metric type {:?}", metric),
        }
    }
}

// reason = "Not yet supported Vector i8"
#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<i8, N> for [i8; N] {
    fn distance_compare(_a: &[i8; N], _b: &[i8; N], _metric: Metric) -> f32 {
        panic!("Not supported VectorType i8")
    }
}

// reason = "Not yet supported Vector u8"
#[allow(clippy::panic)]
impl<const N: usize> FullPrecisionDistance<u8, N> for [u8; N] {
    fn distance_compare(_a: &[u8; N], _b: &[u8; N], _metric: Metric) -> f32 {
        panic!("Not supported VectorType u8")
    }
}

#[cfg(test)]
mod distance_test {
    use super::*;

    #[repr(C, align(32))]
    pub struct F32Slice112([f32; 112]);

    #[repr(C, align(32))]
    pub struct F16Slice112([Half; 112]);

    fn get_turing_test_data() -> (F32Slice112, F32Slice112) {
        let a_slice: [f32; 112] = [
            0.13961786,
            -0.031577103,
            -0.09567415,
            0.06695563,
            -0.1588727,
            0.089852564,
            -0.019837005,
            0.07497972,
            0.010418192,
            -0.054594643,
            0.08613386,
            -0.05103466,
            0.16568437,
            -0.02703799,
            0.00728657,
            -0.15313251,
            0.16462992,
            -0.030570814,
            0.11635703,
            0.23938893,
            0.018022912,
            -0.12646551,
            0.018048918,
            -0.035986554,
            0.031986624,
            -0.015286017,
            0.010117953,
            -0.032691937,
            0.12163067,
            -0.04746277,
            0.010213069,
            -0.043672588,
            -0.099362016,
            0.06599016,
            -0.19397286,
            -0.13285528,
            -0.22040887,
            0.017690737,
            -0.104262285,
            -0.0044555613,
            -0.07383778,
            -0.108652934,
            0.13399786,
            0.054912474,
            0.20181285,
            0.1795591,
            -0.05425621,
            -0.10765217,
            0.1405377,
            -0.14101997,
            -0.12017701,
            0.011565498,
            0.06952187,
            0.060136646,
            0.0023214167,
            0.04204699,
            0.048470616,
            0.17398086,
            0.024218207,
            -0.15626553,
            -0.11291045,
            -0.09688122,
            0.14393932,
            -0.14713104,
            -0.108876854,
            0.035279203,
            -0.05440188,
            0.017205412,
            0.011413814,
            0.04009471,
            0.11070237,
            -0.058998976,
            0.07260045,
            -0.057893746,
            -0.0036240944,
            -0.0064988653,
            -0.13842176,
            -0.023219328,
            0.0035885905,
            -0.0719257,
            -0.21335067,
            0.11415403,
            -0.0059823603,
            0.12091869,
            0.08136634,
            -0.10769281,
            0.024518685,
            0.0009200326,
            -0.11628049,
            0.07448965,
            0.13736208,
            -0.04144517,
            -0.16426727,
            -0.06380103,
            -0.21386267,
            0.022373492,
            -0.05874115,
            0.017314062,
            -0.040344074,
            0.01059176,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let b_slice: [f32; 112] = [
            -0.07209058,
            -0.17755842,
            -0.030627966,
            0.163028,
            -0.2233766,
            0.057412963,
            0.0076995124,
            -0.017121306,
            -0.015759075,
            -0.026947778,
            -0.010282468,
            -0.23968373,
            -0.021486737,
            -0.09903155,
            0.09361805,
            0.0042711576,
            -0.08695552,
            -0.042165346,
            0.064218745,
            -0.06707651,
            0.07846054,
            0.12235762,
            -0.060716823,
            0.18496591,
            -0.13023394,
            0.022469055,
            0.056764495,
            0.07168404,
            -0.08856144,
            -0.15343173,
            0.099879816,
            -0.033529017,
            0.0795304,
            -0.009242254,
            -0.10254546,
            0.13086525,
            -0.101518914,
            -0.1031299,
            -0.056826904,
            0.033196196,
            0.044143833,
            -0.049787212,
            -0.018148342,
            -0.11172959,
            -0.06776237,
            -0.09185828,
            -0.24171598,
            0.05080982,
            -0.0727684,
            0.045031235,
            -0.11363879,
            -0.063389264,
            0.105850354,
            -0.19847773,
            0.08828623,
            -0.087071925,
            0.033512704,
            0.16118294,
            0.14111553,
            0.020884402,
            -0.088860825,
            0.018745849,
            0.047522716,
            -0.03665169,
            0.15726231,
            -0.09930561,
            0.057844743,
            -0.10532736,
            -0.091297254,
            0.067029804,
            0.04153976,
            0.06393326,
            0.054578528,
            0.0038539872,
            0.1023088,
            -0.10653885,
            -0.108500294,
            -0.046606563,
            0.020439683,
            -0.120957725,
            -0.13334097,
            -0.13425854,
            -0.20481694,
            0.07009538,
            0.08660361,
            -0.0096641015,
            0.095316306,
            -0.002898167,
            -0.19680002,
            0.08466311,
            0.04812689,
            -0.028978813,
            0.04780206,
            -0.2001506,
            -0.036866356,
            -0.023720587,
            0.10731964,
            0.05517358,
            -0.09580819,
            0.14595725,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        (F32Slice112(a_slice), F32Slice112(b_slice))
    }

    fn get_turing_test_data_f16() -> (F16Slice112, F16Slice112) {
        let (a_slice, b_slice) = get_turing_test_data();
        let a_data = a_slice.0.iter().map(|x| Half::from_f32(*x));
        let b_data = b_slice.0.iter().map(|x| Half::from_f32(*x));

        (
            F16Slice112(a_data.collect::<Vec<Half>>().try_into().unwrap()),
            F16Slice112(b_data.collect::<Vec<Half>>().try_into().unwrap()),
        )
    }

    use crate::test_util::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dist_l2_float_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data();
        let distance = <[f32; 112] as FullPrecisionDistance<f32, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::L2,
        );

        assert_abs_diff_eq!(
            distance,
            no_vector_compare_f32(&a_slice.0, &b_slice.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_dist_l2_f16_turing() {
        // two vectors are allocated in the contiguous heap memory
        let (a_slice, b_slice) = get_turing_test_data_f16();
        let distance = <[Half; 112] as FullPrecisionDistance<Half, 112>>::distance_compare(
            &a_slice.0,
            &b_slice.0,
            Metric::L2,
        );

        // Note the variance between the full 32 bit precision and the 16 bit precision
        assert_eq!(distance, no_vector_compare_f16(&a_slice.0, &b_slice.0));
    }

    #[test]
    fn distance_test() {
        #[repr(C, align(32))]
        struct Vector32ByteAligned {
            v: [f32; 512],
        }

        // two vectors are allocated in the contiguous heap memory
        let two_vec = Box::new(Vector32ByteAligned {
            v: [
                69.02492, 78.84786, 63.125072, 90.90581, 79.2592, 70.81731, 3.0829668, 33.33287,
                20.777142, 30.147898, 23.681915, 42.553043, 12.602162, 7.3808074, 19.157589,
                65.6791, 76.44677, 76.89124, 86.40756, 84.70118, 87.86142, 16.126896, 5.1277637,
                95.11038, 83.946945, 22.735607, 11.548555, 59.51482, 24.84603, 15.573776, 78.27185,
                71.13179, 38.574017, 80.0228, 13.175261, 62.887978, 15.205181, 18.89392, 96.13162,
                87.55455, 34.179806, 62.920044, 4.9305916, 54.349373, 21.731495, 14.982187,
                40.262867, 20.15214, 36.61963, 72.450806, 55.565, 95.5375, 93.73356, 95.36308,
                66.30762, 58.0397, 18.951357, 67.11702, 43.043316, 30.65622, 99.85361, 2.5889993,
                27.844774, 39.72441, 46.463238, 71.303764, 90.45308, 36.390602, 63.344395,
                26.427078, 35.99528, 82.35505, 32.529175, 23.165905, 74.73179, 9.856939, 59.38126,
                35.714924, 79.81213, 46.704124, 24.47884, 36.01743, 0.46678782, 29.528152,
                1.8980742, 24.68853, 75.58984, 98.72279, 68.62601, 11.890173, 49.49361, 55.45572,
                72.71067, 34.107483, 51.357758, 76.400635, 81.32725, 66.45081, 17.848074,
                62.398876, 94.20444, 2.10886, 17.416393, 64.88253, 29.000723, 62.434315, 53.907238,
                70.51412, 78.70744, 55.181683, 64.45116, 23.419212, 53.68544, 43.506958, 46.89598,
                35.905994, 64.51397, 91.95555, 20.322979, 74.80128, 97.548744, 58.312725, 78.81985,
                31.911612, 14.445949, 49.85094, 70.87396, 40.06766, 7.129991, 78.48008, 75.21636,
                93.623604, 95.95479, 29.571129, 22.721554, 26.73875, 52.075504, 56.783104,
                94.65493, 61.778534, 85.72401, 85.369514, 29.922367, 41.410553, 94.12884,
                80.276855, 55.604828, 54.70947, 74.07216, 44.61955, 31.38113, 68.48596, 34.56782,
                14.424729, 48.204506, 9.675444, 32.01946, 92.32695, 36.292683, 78.31955, 98.05327,
                14.343918, 46.017002, 95.90888, 82.63626, 16.873539, 3.698051, 7.8042626,
                64.194405, 96.71023, 67.93692, 21.618402, 51.92182, 22.834194, 61.56986, 19.749891,
                55.31206, 38.29552, 67.57593, 67.145836, 38.92673, 94.95708, 72.38746, 90.70901,
                69.43995, 9.394085, 31.646872, 88.20112, 9.134722, 99.98214, 5.423498, 41.51995,
                76.94409, 77.373276, 3.2966614, 9.611201, 57.231106, 30.747868, 76.10228, 91.98308,
                70.893585, 0.9067178, 43.96515, 16.321218, 27.734184, 83.271835, 88.23312,
                87.16445, 5.556643, 15.627432, 58.547127, 93.6459, 40.539192, 49.124157, 91.13276,
                57.485855, 8.827019, 4.9690843, 46.511234, 53.91469, 97.71925, 20.135271,
                23.353004, 70.92099, 93.38748, 87.520134, 51.684677, 29.89813, 9.110392, 65.809204,
                34.16554, 93.398605, 84.58669, 96.409645, 9.876037, 94.767784, 99.21523, 1.9330144,
                94.92429, 75.12728, 17.218828, 97.89164, 35.476578, 77.629456, 69.573746,
                40.200542, 42.117836, 5.861628, 75.45282, 82.73633, 0.98086596, 77.24894,
                11.248695, 61.070026, 52.692616, 80.5449, 80.76036, 29.270136, 67.60252, 48.782394,
                95.18851, 83.47162, 52.068756, 46.66002, 90.12216, 15.515327, 33.694042, 96.963036,
                73.49627, 62.805485, 44.715607, 59.98627, 3.8921833, 37.565327, 29.69184,
                39.429665, 83.46899, 44.286453, 21.54851, 56.096413, 18.169249, 5.214751,
                14.691341, 99.779335, 26.32643, 67.69903, 36.41243, 67.27333, 12.157213, 96.18984,
                2.438283, 78.14289, 0.14715195, 98.769, 53.649532, 21.615898, 39.657497, 95.45616,
                18.578386, 71.47976, 22.348118, 17.85519, 6.3717127, 62.176777, 22.033644,
                23.178005, 79.44858, 89.70233, 37.21273, 71.86182, 21.284317, 52.908623, 30.095518,
                63.64478, 77.55823, 80.04871, 15.133011, 30.439043, 70.16561, 4.4014096, 89.28944,
                26.29093, 46.827854, 11.764729, 61.887516, 47.774887, 57.19503, 59.444664,
                28.592825, 98.70386, 1.2497544, 82.28431, 46.76423, 83.746124, 53.032673, 86.53457,
                99.42168, 90.184, 92.27852, 9.059965, 71.75723, 70.45299, 10.924053, 68.329704,
                77.27232, 6.677854, 75.63629, 57.370533, 17.09031, 10.554659, 99.56178, 37.53221,
                72.311104, 75.7565, 65.2042, 36.096478, 64.69502, 38.88497, 64.33723, 84.87812,
                66.84958, 8.508932, 79.134, 83.431015, 66.72124, 61.801838, 64.30524, 37.194263,
                77.94725, 89.705185, 23.643505, 19.505919, 48.40264, 43.01083, 21.171177,
                18.717121, 10.805857, 69.66983, 77.85261, 57.323063, 3.28964, 38.758026, 5.349946,
                7.46572, 57.485138, 30.822384, 33.9411, 95.53746, 65.57723, 42.1077, 28.591347,
                11.917269, 5.031073, 31.835615, 19.34116, 85.71027, 87.4516, 1.3798475, 70.70583,
                51.988052, 45.217144, 14.308596, 54.557167, 86.18323, 79.13666, 76.866745,
                46.010685, 79.739235, 44.667603, 39.36416, 72.605896, 73.83187, 13.137412,
                6.7911267, 63.952374, 10.082436, 86.00318, 99.760376, 92.84948, 63.786434,
                3.4429908, 18.244314, 75.65299, 14.964747, 70.126366, 80.89449, 91.266655,
                96.58798, 46.439327, 38.253975, 87.31036, 21.093178, 37.19671, 58.28973, 9.75231,
                12.350321, 25.75115, 87.65073, 53.610504, 36.850048, 18.66356, 94.48941, 83.71898,
                44.49315, 44.186737, 19.360733, 84.365974, 46.76272, 44.924366, 50.279808,
                54.868866, 91.33004, 18.683397, 75.13282, 15.070831, 47.04839, 53.780903,
                26.911152, 74.65651, 57.659935, 25.604189, 37.235474, 65.39667, 53.952206,
                40.37131, 59.173275, 96.00756, 54.591274, 10.787476, 69.51549, 31.970142,
                25.408005, 55.972492, 85.01888, 97.48981, 91.006134, 28.98619, 97.151276,
                34.388496, 47.498177, 11.985874, 64.73775, 33.877014, 13.370312, 34.79146,
                86.19321, 15.019405, 94.07832, 93.50433, 60.168625, 50.95409, 38.27827, 47.458614,
                32.83715, 69.54998, 69.0361, 84.1418, 34.270298, 74.23852, 70.707466, 78.59845,
                9.651399, 24.186779, 58.255756, 53.72362, 92.46477, 97.75528, 20.257462, 30.122698,
                50.41517, 28.156603, 42.644154,
            ],
        });

        let distance = compare::<f32, 256>(256, Metric::L2, &two_vec.v);

        assert_eq!(distance, 429141.2);
    }

    fn compare<T, const N: usize>(dim: usize, metric: Metric, v: &[f32]) -> f32
    where
        for<'a> [T; N]: FullPrecisionDistance<T, N>,
    {
        let a_ptr = v.as_ptr();
        let b_ptr = unsafe { a_ptr.add(dim) };

        let a_ref =
            <&[f32; N]>::try_from(unsafe { std::slice::from_raw_parts(a_ptr, dim) }).unwrap();
        let b_ref =
            <&[f32; N]>::try_from(unsafe { std::slice::from_raw_parts(b_ptr, dim) }).unwrap();

        <[f32; N]>::distance_compare(a_ref, b_ref, metric)
    }
}
