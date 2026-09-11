#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Helper buffers: normalized logits for a batch, and the gradient of theta
    float *Z = new float[batch * k];
    float *grad = new float[n * k];

    for (size_t start = 0; start < m; start += batch) {
        size_t b = (m - start < batch) ? (m - start) : batch;  // last batch may be smaller

        // Z = normalize(exp(X_b * theta)) - I_y,  shape (b, k)
        for (size_t i = 0; i < b; i++) {
            const float *x = X + (start + i) * n;
            float *z = Z + i * k;

            for (size_t j = 0; j < k; j++) {
                float s = 0.0f;
                for (size_t l = 0; l < n; l++) {
                    s += x[l] * theta[l * k + j];
                }
                z[j] = s;
            }

            // softmax (subtract max for numerical stability)
            float zmax = z[0];
            for (size_t j = 1; j < k; j++) if (z[j] > zmax) zmax = z[j];
            float total = 0.0f;
            for (size_t j = 0; j < k; j++) {
                z[j] = std::exp(z[j] - zmax);
                total += z[j];
            }
            for (size_t j = 0; j < k; j++) z[j] /= total;

            z[y[start + i]] -= 1.0f;
        }

        // grad = X_b^T * Z / b,  shape (n, k)
        for (size_t idx = 0; idx < n * k; idx++) grad[idx] = 0.0f;
        for (size_t i = 0; i < b; i++) {
            const float *x = X + (start + i) * n;
            const float *z = Z + i * k;
            for (size_t l = 0; l < n; l++) {
                float xl = x[l];
                if (xl == 0.0f) continue;  // MNIST is sparse; skip zero pixels
                for (size_t j = 0; j < k; j++) {
                    grad[l * k + j] += xl * z[j];
                }
            }
        }

        // theta -= lr * grad
        float scale = lr / static_cast<float>(b);
        for (size_t idx = 0; idx < n * k; idx++) {
            theta[idx] -= scale * grad[idx];
        }
    }

    delete[] Z;
    delete[] grad;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}