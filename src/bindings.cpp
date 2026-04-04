#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include "IndexIVF.h"
#include "iostream"

namespace py = pybind11;

// "vecmini" is the name of the module you will type in python-> 'import vecmini'
PYBIND11_MODULE(vecmini, m) {
    m.doc() = "Vecmini: A mini custom IVF Vector Database with Metadata Filtering";

    py::class_<IndexIVF>(m, "IndexIVF")
        // Expose the constructor
        .def(py::init<int, int>(), py::arg("d"), py::arg("nbucket"))
        
        // Expose train() - Unchanged
        .def("train", [](IndexIVF &self, int n, py::array_t<float, py::array::c_style | py::array::forcecast> x) {
            py::buffer_info buf = x.request();
            self.train(n, (const float *)buf.ptr);
        }, py::arg("n"), py::arg("x"))
        
        // Expose add() - UPDATED FOR PARALLEL ARRAYS (xids)
        .def("add", [](IndexIVF &self, int n, 
                       py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                       py::array_t<uint64_t, py::array::c_style | py::array::forcecast> xids) {
            
            py::buffer_info buf_x = x.request();
            py::buffer_info buf_xids = xids.request();
            
            self.add(n, (const float *)buf_x.ptr, (const uint64_t *)buf_xids.ptr);
        }, py::arg("n"), py::arg("x"), py::arg("xids"))
        
        // Expose search() - UPDATED FOR NPROBE AND BITMASK
        .def("search", [](IndexIVF &self, int n, 
                          py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                          int k, int nprobe, py::object bitmask) {
            
            py::buffer_info buf_x = x.request();
            
            // Empty arrays to hold the answers for Python
            py::array_t<float> distances({n, k});
            py::array_t<int> labels({n, k});

            const uint8_t* bitmask_ptr = nullptr;
            py::array_t<uint8_t> bitmask_arr; 
            
            if (!bitmask.is_none()) {
                bitmask_arr = bitmask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
                bitmask_ptr = (const uint8_t*)bitmask_arr.request().ptr;
                std::cout<<"recieved bitmask , *pointer address->" <<(void*)bitmask_ptr<<"\n";
            } else {
                std::cout<<"recieved NONE\n";
            }
            
            // THE FIX: Use mutable_data() directly!
            self.search(n, (const float *)buf_x.ptr, k, nprobe, bitmask_ptr, 
                        distances.mutable_data(), labels.mutable_data());
            
            return py::make_tuple(distances, labels);
        }, py::arg("n"), py::arg("x"), py::arg("k"), py::arg("nprobe"), py::arg("bitmask"));
}