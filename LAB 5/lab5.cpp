
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

class Vector
{
public:
    explicit Vector(double x = 0, double y = 0, double z = 0)
    {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm2() const
    {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm() const
    {
        return sqrt(norm2());
    }
    void normalize()
    {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
        data[2] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double &operator[](int i) { return data[i]; };
    double data[3];
};

Vector operator+(const Vector &a, const Vector &b)
{
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector &a, const Vector &b)
{
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator-(const Vector &a)
{
	return Vector(-a[0], -a[1], -a[2]);
}
Vector operator*(const double a, const Vector &b)
{
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector &a, const double b)
{
    return Vector(a[0] * b, a[1] * b, a[2] * b);
}
Vector operator*(const Vector &a, const Vector &b)
{
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
Vector operator/(const Vector &a, const double b)
{
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
Vector operator/(const Vector &a, const Vector &b)
{
	return Vector(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
}
double dot(const Vector &a, const Vector &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector &a, const Vector &b)
{
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

double cross2(const Vector &a, const Vector &b)
{
    return a[0] * b[1] - a[1] * b[0];
}

Vector random_direction() {
    double r1 = ((double) rand() / (RAND_MAX));
    double r2 = ((double) rand() / (RAND_MAX));
    double x = cos(2 * M_PI * r1) * sqrt(r2 * (1 - r2));
    double y = sin(2 * M_PI * r1) * sqrt(r2 * (1 - r2));
    double z = 1 - 2 * r2;
    return Vector(x, y, z);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Error: wrong arguments.\n");
        exit(1);
    }
    const char *input_image_filename = argv[1];
    const char *model_image_filename = argv[2];
    size_t iterations = atoi(argv[3]);

    int input_W, input_H, input_channels;
    unsigned char *input_image = stbi_load(input_image_filename, &input_W, &input_H, &input_channels, 0);
    if (input_image == NULL) {
        printf("Error: failed to load the input image.\n");
        exit(1);
    }

    int model_W, model_H, model_channels;
    unsigned char *model_image = stbi_load(model_image_filename, &model_W, &model_H, &model_channels, 0);
    if (model_image == NULL) {
        printf("Error: failed to load the model image.\n");
        exit(1);
    }

    if (model_W != input_W || model_H != input_H || model_channels != input_channels) {
        printf("Error: input and model must have same dimensions and channels.\n");
        exit(1);
    }

    size_t total_pixels = input_W * input_H;
    std::vector<std::pair<int, int>> projI(total_pixels);
    std::vector<std::pair<int, int>> projM(total_pixels);
    Vector pixel, model_pixel, v;
    
    for (size_t iter = 0; iter < iterations; iter++) {
        v = random_direction();

        for (size_t i = 0; i < total_pixels; i++) {
            unsigned char *I = input_image + input_channels * i;
            unsigned char *M = model_image + model_channels * i;
            pixel = Vector(*I, *(I + 1), *(I + 2));
            model_pixel = Vector(*M, *(M + 1), *(M + 2));
            projI[i] = std::pair<int, int>(dot(pixel, v), i);
            projM[i] = std::pair<int, int>(dot(model_pixel, v), i);
        }

        std::sort(projI.begin(), projI.end());
        std::sort(projM.begin(), projM.end());

        for (size_t i = 0; i < total_pixels; i++) {
            int permutation_index = projI[i].second;
            unsigned char *I = input_image + input_channels * permutation_index;
            pixel = Vector(*I, *(I + 1), *(I + 2)) + (projM[i].first - projI[i].first)*v;
            *I = pixel[0];
            *(I + 1) = pixel[1];
            *(I + 2) = pixel[2];
        }
    }

    stbi_write_png("output.png", input_W, input_H, input_channels, &input_image[0], 0);
    
    return 0;
}