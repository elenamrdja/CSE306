#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <list>
#include <math.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <random>
#include <sstream>

#include "lbfgs.c"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define _CRT_SECURE_NO_WARNINGS 1

bool g_isFluid;


class Vector
{
public:
    explicit Vector(double x = 0, double y = 0, double z = 0)
    {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm() const
    {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm_sqrt() const
    {
        return sqrt(norm());
    }
    void normalize()
    {
        double n = norm_sqrt();
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

double cross(const Vector &a, const Vector &b)
{
    return a[0] * b[1] - a[1] * b[0];
}




class Polygon
{
public:
    std::vector<Vector> coords;

    double surface() const;

	Vector centroid() 
    {
		Vector C(0, 0, 0);
		double S = surface();
        int N = coords.size();
		if (S== 0)
			if (N != 0)
				return coords[0];
			else
				return Vector(0, 0, 0);
		for (int i = 0; i < N; i++) 
		{
			int j = (i+1) % N;
			C = C + (coords[i] + coords[j]) * cross(coords[i], coords[j]);
		}
		return -1 * C / (6 * S);
	}
};

double computePolygonsurface(const Polygon& polygon)
{
    const std::vector<Vector>& coords = polygon.coords;
    double surface = 0.0;
    int vertexCount = coords.size();
    
    if (vertexCount < 2)
        return 0;
    
    for (int i = 0; i < vertexCount; i++)
    {
        int nextIndex = (i + 1) % vertexCount;
        double x1 = coords[i][0];
        double y1 = coords[i][1];
        double x2 = coords[nextIndex][0];
        double y2 = coords[nextIndex][1];
        
        surface += (x1 * y2) - (x2 * y1);
    }
    
    return std::abs(surface/2.0);
}


double Polygon::surface() const
{
    return computePolygonsurface(*this);
}


int sgn(double x) {
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

void save_svg(const std::vector<Polygon> &polygons, std::string filename, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+"); 
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i=0; i<polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \""); 
        for (int j = 0; j < polygons[i].coords.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].coords[j][0] * 1000), (1000 - polygons[i].coords[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }
    fprintf(f, "</svg>\n");
    fclose(f);
}

void save_svg_animated(const std::vector<Polygon> &polygons, std::string filename, int frameid, int nbframes) {
    FILE* f;
    if (frameid == 0) {
        f = fopen(filename.c_str(), "w+");
        fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
        fprintf(f, "<g>\n");
    } else {
        f = fopen(filename.c_str(), "a+");
    }
    fprintf(f, "<g>\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<polygon points = \""); 
        for (int j = 0; j < polygons[i].coords.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].coords[j][0] * 1000), (1000-polygons[i].coords[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"none\" stroke = \"black\"/>\n");
    }
    fprintf(f, "<animate\n");
    fprintf(f, "	id = \"frame%u\"\n", frameid);
    fprintf(f, "	attributeName = \"display\"\n");
    fprintf(f, "	values = \"");
    for (int j = 0; j < nbframes; j++) {
        if (frameid == j) {
            fprintf(f, "inline");
        } else {
            fprintf(f, "none");
        }
        fprintf(f, ";");
    }
    fprintf(f, "none\"\n	keyTimes = \"");
    for (int j = 0; j < nbframes; j++) {
        fprintf(f, "%2.3f", j / (double)(nbframes));
        fprintf(f, ";");
    }
    fprintf(f, "1\"\n	dur = \"5s\"\n");
    fprintf(f, "	begin = \"0s\"\n");
    fprintf(f, "	repeatCount = \"indefinite\"/>\n");
    fprintf(f, "</g>\n");
    if (frameid == nbframes - 1) {
        fprintf(f, "</g>\n");
        fprintf(f, "</svg>\n");
    }
    fclose(f);
}

void save_frame(const std::vector<Polygon> &cells, std::string filename, int frameid = 0) {
    int W = 1000, H = 1000;
    std::vector<unsigned char> image(W*H * 3, 255);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < cells.size(); i++) {

        double bminx = 1E9, bminy = 1E9, bmaxx = -1E9, bmaxy = -1E9;
        for (int j = 0; j < cells[i].coords.size(); j++) {
            bminx = std::min(bminx, cells[i].coords[j][0]);
            bminy = std::min(bminy, cells[i].coords[j][1]);
            bmaxx = std::max(bmaxx, cells[i].coords[j][0]);
            bmaxy = std::max(bmaxy, cells[i].coords[j][1]);
        }
        bminx = std::min(W-1., std::max(0., W * bminx));
        bminy = std::min(H-1., std::max(0., H * bminy));
        bmaxx = std::max(W-1., std::max(0., W * bmaxx));
        bmaxy = std::max(H-1., std::max(0., H * bmaxy));

        for (int y = bminy; y < bmaxy; y++) {
            for (int x = bminx; x < bmaxx; x++) {
                int prevSign = 0;
                bool isInside = true;
                double mindistEdge = 1E9;
                for (int j = 0; j < cells[i].coords.size(); j++) {
                    double x0 = cells[i].coords[j][0] * W;
                    double y0 = cells[i].coords[j][1] * H;
                    double x1 = cells[i].coords[(j + 1) % cells[i].coords.size()][0] * W;
                    double y1 = cells[i].coords[(j + 1) % cells[i].coords.size()][1] * H;
                    double det = (x - x0)*(y1-y0) - (y - y0)*(x1-x0);
                    int sign = sgn(det);
                    if (prevSign == 0) prevSign = sign; else
                        if (sign == 0) sign = prevSign; else
                        if (sign != prevSign) {
                            isInside = false;
                            break;
                        }
                    prevSign = sign;
                    double edgeLen = sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0));
                    double distEdge = std::abs(det)/ edgeLen;
                    double dotp = (x - x0)*(x1 - x0) + (y - y0)*(y1 - y0);
                    if (dotp<0 || dotp>edgeLen*edgeLen) distEdge = 1E9;
                    mindistEdge = std::min(mindistEdge, distEdge);
                }
                if (isInside) {
                        image[((H - y - 1)*W + x) * 3] = 0;
                        image[((H - y - 1)*W + x) * 3 + 1] = 0;
                        image[((H - y - 1)*W + x) * 3 + 2] = 255;
                    if (mindistEdge <= 2) {
                        image[((H - y - 1)*W + x) * 3] = 0;
                        image[((H - y - 1)*W + x) * 3 + 1] = 0;
                        image[((H - y - 1)*W + x) * 3 + 2] = 0;
                    }

                }
                
            }
        }
    }
    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, &image[0], 0);
}

Polygon get_clipping(Polygon& cell, int i, int j, const Vector* points, const double* weights)
{
	Polygon result;
	int N = cell.coords.size();
	for (int k = 0; k < N; k++) 
	{
		Vector& A = cell.coords[((k-1)>=0)?(k-1):(N-1)];
		Vector& B = cell.coords[k];

		Vector M = (points[i] + points[j]) / 2;
		M = M + (weights[i] - weights[j]) / (2 * (points[i] - points[j]).norm()) * (points[j] - points[i]);
		
        double t = dot(M-A, points[j] - points[i]) / dot(B-A, points[j] - points[i]);
		Vector P = A + t * (B - A);

		if (((B - points[i]).norm() - weights[i]) <= ((B - points[j]).norm() - weights[j]))
		{
			if (((A - points[i]).norm() - weights[i]) > ((A - points[j]).norm() - weights[j]))
			{
				result.coords.push_back(P);
			}
			result.coords.push_back(B);
		}
		else if (((A - points[i]).norm() - weights[i]) <= ((A - points[j]).norm() - weights[j]))
        {
            result.coords.push_back(P);
        }
		
	}
	return result;
}


Polygon get_clipped(const Polygon& poly, const Polygon& clip)
{
    Polygon result = poly;
    int N = clip.coords.size();
    for (int i = 0; i < N; i++)
    {
        Vector P1 = clip.coords[i];
        Vector P2 = clip.coords[(i + 1) % N];
        Vector u = P1;
        Vector n(P2[1] - P1[1], -P2[0] + P1[0]);
        int M = result.coords.size();
        Polygon tmp;
        for (int k = 0; k < M; k++)
        {
            Vector& A = result.coords[((k - 1) >= 0) ? (k - 1) : (M - 1)];
            Vector& B = result.coords[k];

            double t = dot(u - A, n) / dot(B - A, n);
            Vector P = A + t * (B - A);

            if (dot(u - B, n) <= 0)
            {
                if (dot(u - A, n) > 0)
                {
                    tmp.coords.push_back(P);
                }
                tmp.coords.push_back(B);
            }
            else if (dot(u - A, n) <= 0)
            {
                tmp.coords.push_back(P);
            }
        }
        result = tmp;
    }
    return result;
}


std::vector<Polygon> compute_diagram(const Vector* points, const double* weights, int N, bool isFluid)
{
    std::vector<Polygon> diagram(isFluid ? N - 1 : N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (isFluid ? N - 1 : N); i++)
    {
        Polygon cell;
        cell.coords.push_back(Vector(0, 0, 0));
        cell.coords.push_back(Vector(0, 1, 0));
        cell.coords.push_back(Vector(1, 1, 0));
        cell.coords.push_back(Vector(1, 0, 0));

        for (int j = 0; j < N; j++)
        {
            cell = get_clipping(cell, i, j, points, weights);
        }

        if (isFluid)
        {
            Polygon disk;
            disk.coords.resize(200);
            double radius = sqrt(weights[i] - weights[N - 1]);
            for (int j = 0; j < 200; j++)
            {
                disk.coords[j][0] = cos(j / 200. * 2 * M_PI) * radius + points[i][0];
                disk.coords[j][1] = -sin(j / 200. * 2 * M_PI) * radius + points[i][1];
                disk.coords[j][2] = 0;
            }

            cell = get_clipped(cell, disk);
        }

        diagram[i] = cell;
    }

    return diagram;
}

static lbfgsfloatval_t lbfgs_optimize(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int N,
    const lbfgsfloatval_t step
    )
{
    int i;
    lbfgsfloatval_t fx = 0.0;

    Vector* points = static_cast<Vector*>(instance);

    std::vector<Polygon> diagram = compute_diagram(&points[0], &x[0], N, false);

    double lambda = 1.0 / N;
    for (i = 0; i < N; i++)
    {
        double surface = computePolygonsurface(diagram[i]);
        g[i] = -(lambda - surface);

        double A = 0;
        int M = diagram[i].coords.size();
        if (M < 3)
            continue;

        for (int j = 1; j < M - 1; j++)
        {
            double temp = std::abs(0.5 * cross(diagram[i].coords[j] - diagram[i].coords[0], diagram[i].coords[j + 1] - diagram[i].coords[0]));
            Vector C[3] = { diagram[i].coords[0], diagram[i].coords[j], diagram[i].coords[j + 1] };
            for (int p = 0; p < 3; p++)
            {
                for (int q = p; q < 3; q++)
                {
                    A += temp / 6 * dot(C[p] - points[i], C[q] - points[i]);
                }
            }
        }

        fx += -(A - x[i] * surface + lambda * x[i]);
    }

    return fx;
}


static lbfgsfloatval_t lbfgs_optimize_fluid(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    lbfgsfloatval_t fx = 0.0;

    Vector* points = static_cast<Vector*>(instance);

    std::vector<Polygon> diagram = compute_diagram(&points[0], &x[0], n, true);

    double fraction_fluid = 0.4;
    double fraction_air = 1 - fraction_fluid;
    double lambda = fraction_fluid / (n - 1);
    double sum_fluid_surface = 0;

    for (int i = 0; i < n - 1; i++)
    {
        double surface = computePolygonsurface(diagram[i]);
        sum_fluid_surface += surface;
        g[i] = -(lambda - surface);

        double A = 0;
        int M = diagram[i].coords.size();
        if (M < 3)
            continue;

        for (int j = 1; j < M - 1; j++)
        {
            double temp = std::abs(0.5 * cross(diagram[i].coords[j] - diagram[i].coords[0], diagram[i].coords[j + 1] - diagram[i].coords[0]));
            Vector C[3] = { diagram[i].coords[0], diagram[i].coords[j], diagram[i].coords[j + 1] };
            for (int p = 0; p < 3; p++)
            {
                for (int q = p; q < 3; q++)
                {
                    A += temp / 6 * dot(C[p] - points[i], C[q] - points[i]);
                }
            }
        }

        fx += -(A - x[i] * surface + lambda * x[i]);
    }

    double air_surface = 1 - sum_fluid_surface;
    g[n - 1] = -(fraction_air - air_surface);
    fx += -(-x[n - 1] * air_surface + fraction_air * x[n - 1]);

    return fx;
}

// LAB 9 - TUTTE EMBEDDING
std::vector<Vector> tutte_embedding(
    std::vector<Vector> points,
    const std::vector<std::vector<int> >& adj,
    const std::vector<int>& bound,
    int iter
) {
    int n = bound.size();
    double s = 0;
    for (int i = 0; i < n; i++) {
        s = s + sqrt((points[bound[i]] - points[bound[(i + 1) % n]]).norm());
    }
    
    double cs = 0;
    for (int i = 0; i < n; i++) {

        double theta = 2 * M_PI * cs / s;
        points[bound[i]] = Vector(cos(theta), sin(theta), 0);
        cs = cs + sqrt((points[bound[i]] - points[bound[(i + 1) % n]]).norm());
    }
    for (int j = 0; j < iter; j++) 
	{

        std::vector<Vector> tmp = points;
        for (uint i = 0; i < n; i++)
		 {
            tmp[i] = Vector(0., 0., 0.);
            size_t k_size = adj[i].size();


            for (int k = 0; k < k_size; k++) 
			{
                tmp[i] = tmp[i] + points[adj[i][k]];
            }
            tmp[i] = tmp[i] / k_size;
        }
        for (int i = 0; i < n; i++) 
		{
            tmp[bound[i]] = points[bound[i]];
        }
        points = tmp;
    }
    return points;
}



int main()
{
	bool fluid = true;
    int N = 200;

    std::vector<Vector> points(N);
    std::vector<Vector> velocity(N);
    std::vector<double> weights(N, 0);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            points[i][j] = rand() / (double)RAND_MAX;
            velocity[i][j] = 0;
        }
        velocity[i][2] = 0;
        points[i][2] = 0;
        weights[i] = 0.05;
    }

    for (int t = 0; t < 200; t++) {
        double m = 200;
        double epsilon = 0.004 * 0.004;
        double dt = 0.002;

        int N = points.size() + 1;
        double fx;

        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;

        if (fluid) {
            int ret = lbfgs(N, &weights[0], &fx, lbfgs_optimize_fluid, NULL, &points[0], &param);
            std::cout << ret << std::endl;
        } 
		else{
            int ret = lbfgs(N, &weights[0], &fx, lbfgs_optimize, NULL, &points[0], &param);
            std::cout << ret << std::endl;
        }

        std::vector<Polygon> cells = compute_diagram(&points[0], &weights[0], N, fluid);
        save_frame(cells, "images/image_", t);

        std::vector<Polygon> diagram = compute_diagram(&points[0], &weights[0], N, fluid);

        for (int i = 0; i < N - 1; i++)
        {
            Vector F_spring = 1 / epsilon * (diagram[i].centroid() - points[i]);
            Vector F = F_spring + Vector(0, -9.81) * m;
            velocity[i] = velocity[i] + dt * F / m;
            points[i] = points[i] + dt * velocity[i];
            if (points[i][0] < 0) points[i][0] = -points[i][0];
            if (points[i][1] < 0) points[i][1] = -points[i][1];
            if (points[i][0] >= 1) points[i][0] = 2 - points[i][0];
            if (points[i][1] >= 1) points[i][1] = 2 - points[i][1];
        }
    }

    return 0;
}
