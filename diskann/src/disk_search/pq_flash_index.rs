use vector::Metric;

use crate::disk_search::aligned_file_reader::AlignedFileReader;

pub struct PQFlashIndex<T, LabelT> {
    fileReader: &Box<dyn AlignedFileReader>,
    metric_to_invoke: Metric,
}

impl<T, LabelT> PQFlashIndex<T, LabelT> {
pub fn new(fileReader: &Box<dyn AlignedFileReader>, m: Metric) -> Self
{
    let metric_to_invoke = m;
    if (m == Metric::L2 || m == Metric::Cosine)
    {
        if (std::is_floating_point<T>::value)
        {
            // Since data is floating point, we assume that it has been appropriately pre-processed 
                             // (normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we 
                             // shall invoke an l2 distance function.
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            // WARNING: Cannot normalize integral data types This may result in erroneous results or poor recall
            // Consider using L2 distance with integral data types
        }
    }

    //this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    //this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));

    Self {
        fileReader,
        metric_to_invoke,
    }
}
}
