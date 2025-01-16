#include "../exercise.h"
#include <cstring>  // std::memcpy

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    // 构造函数，计算总大小并分配内存
    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        for (int i = 0; i < 4; ++i) {
            shape[i] = shape_[i];  // 将 shape_ 复制到成员变量 shape
            size *= shape_[i];     // 计算总大小
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));  // 复制数据
    }

    // 析构函数，释放内存
    ~Tensor4D() {
        delete[] data;
    }

    // 禁止复制和移动构造函数
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 加法运算符实现
    Tensor4D &operator+=(Tensor4D const &others) {
        // 检查形状是否满足广播要求
        for (int i = 0; i < 4; ++i) {
            if (shape[i] != others.shape[i] && others.shape[i] != 1) {
                // 如果维度不相同且 `others` 在该维度的大小不为 1，报错
                throw std::invalid_argument("Shape mismatch for broadcasting.");
            }
        }

        // 进行加法操作，根据广播规则
        for (unsigned int i = 0; i < shape[0]; ++i) {
            for (unsigned int j = 0; j < shape[1]; ++j) {
                for (unsigned int k = 0; k < shape[2]; ++k) {
                    for (unsigned int l = 0; l < shape[3]; ++l) {
                        // 计算广播后的索引
                        unsigned int other_i = (others.shape[0] == 1 ? 0 : i);
                        unsigned int other_j = (others.shape[1] == 1 ? 0 : j);
                        unsigned int other_k = (others.shape[2] == 1 ? 0 : k);
                        unsigned int other_l = (others.shape[3] == 1 ? 0 : l);

                        // 执行加法
                        data[(i * shape[1] + j) * shape[2] * shape[3] + k * shape[3] + l] += 
                            others.data[(other_i * others.shape[1] + other_j) * others.shape[2] * others.shape[3] + other_k * others.shape[3] + other_l];
                    }
                }
            }
        }

        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D<int>(shape, data);
        auto t1 = Tensor4D<int>(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D<float>(s0, d0);
        auto t1 = Tensor4D<float>(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D<double>(s0, d0);
        auto t1 = Tensor4D<double>(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
