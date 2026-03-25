#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include "IndexIVF.h"

namespace py = pybind11;

// "vecmini" is the name of the module you will type in python-> 'import vecmini'
PYBIND11_MODULE(vecmini, m) {
    m.doc() = "Vecmini: A mini custom IVF Vector Database";

    py::class_<IndexIVF>(m, "IndexIVF")
        //expose the constructor
        .def(py::init<int, int>(), py::arg("d"), py::arg("nbucket"))
        
        //expose train()
        .def("train", [](IndexIVF &self, int n, py::array_t<float> x) {
            py::buffer_info buf = x.request();
            self.train(n, (const float *)buf.ptr);
        }, py::arg("n"), py::arg("x"))
        
        //expose add()
        .def("add", [](IndexIVF &self, int n, py::array_t<float> x) {
            py::buffer_info buf = x.request();
            self.add(n, (const float *)buf.ptr);
        }, py::arg("n"), py::arg("x"))
        
        //expose search()
        .def("search", [](IndexIVF &self, int n, py::array_t<float> x, int k) {
            py::buffer_info buf_x = x.request();
            
            //empty arrays to hold the answers
            py::array_t<float> distances(n * k);
            py::array_t<int> labels(n * k);
            
            py::buffer_info buf_dist = distances.request();
            py::buffer_info buf_labels = labels.request();
            
            //run search
            self.search(n, (const float *)buf_x.ptr, k, (float *)buf_dist.ptr, (int *)buf_labels.ptr);
            
            //return the answers as a Python tuple->(distances,labels)
            return py::make_tuple(distances, labels);
        }, py::arg("n"), py::arg("x"), py::arg("k"));
}