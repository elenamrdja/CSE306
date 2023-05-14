#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <list>
#include <math.h>
#include <vector>
#include <string>
#include <stdio.h>
//#include <bits/stdc++.h>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define _CRT_SECURE_NO_WARNINGS 1
    
static std::default_random_engine engine(10);
static std::uniform_real_distribution<double> uniform (0 ,1);

//////////////////////////


class Vector
{
public:
    explicit Vector(double x = 0, double y = 0, double z = 0)
    {
        coords[0] = x;
        coords[1] = y;
        coords[2] = z;
    }
    double norm() const
    {
        double n = std::abs(coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2]);
        return sqrt(n);
    }
    void normalization()
    {
        double nm = norm();
        coords[0]/=nm;
        coords[1]/=nm;
        coords[2]/=nm;
    }
    double operator[](int i) const { return coords[i]; };
    double &operator[](int i) { return coords[i]; };
    double coords[3];
};

struct Intersection
{
    Vector P;
    Vector N;
    bool flag = false;
    double t;
    int id = -1;
};

struct Lighting
{
    Vector position;
    double intensity;
};

// ray class
class Ray
{
public:
    Vector origin;
    Vector direction;
    Ray(Vector o, Vector u) {
        origin = o;
        direction= u;
    }
};


Vector operator+(const Vector &a, const Vector &b)
{
    return Vector(a[0]+b[0], a[1]+b[1],a[2]+b[2]);
}

Vector operator-(const Vector &a, const Vector &b)
{
    return Vector(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}


Vector operator*(const Vector &a, const Vector &b)
{
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
Vector operator*(const double a, const Vector &b)
{
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector &a, const double b)
{
    return Vector(a[0] * b, a[1] * b, a[2] * b);
}


Vector operator/(const Vector &a, const double b)
{
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
Vector operator/(const Vector &a, const Vector &b)
{
	return Vector(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
}


Vector operator-(const Vector &a)
{
	return Vector(-a[0], -a[1], -a[2]);
}

double dot(const Vector &a, const Vector &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector &a, const Vector &b)
{
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}



Vector random_cos(const Vector& N) 
{
    double r1 = uniform(engine);
    double r2 =uniform(engine);
    double x =sqrt(1 - r2)*cos(2 * M_PI * r1) ;
    double y =sqrt(1 - r2)*sin(2 * M_PI * r1);
    double z = sqrt(r2);

    Vector T1;
    if (abs(N[1]) <= std::min(abs(N[0]), abs(N[2])))
    {
        T1 = Vector(N[2], 0, -N[1]);
    }
    else if (abs(N[0]) <= std::min(abs(N[1]), abs(N[2])))
    {
        T1 = Vector(0,N[2],-N[1]);
    }
    else
    {
        T1 = Vector(N[1], -N[0], 0);
    }
    
    T1.normalization();
    Vector T2 = cross(N, T1);
    return x * T1 + y * T2 + z * N;
}

// Geometry class
class Geometry
{
public:
    virtual Intersection intersect(Ray &ray) = 0;
    Vector albedo;
    bool mirror;
    bool transparent;
    bool hallow;
    Vector C;
    double R;
    double scale;
    Vector translation;
};


// Sphere class
class Sphere : public Geometry
{
public:
    Sphere(Vector Center, Vector albedo_temp, double Radius, bool mirror_flag = false, bool transparent_flag = false, bool hallow_flag = false)
    {
        C = Center;
        R = Radius;
        albedo = albedo_temp;
        mirror = mirror_flag;
        transparent = transparent_flag;
        hallow = hallow_flag ;
    }

    Intersection intersect(Ray &ray)
    {
        Intersection intersection;
        Vector CO = ray.origin - C;
        double b = dot(ray.direction, CO);
        double delta = b * b + R * R - (CO.norm() * CO.norm());

        if (delta < 0)
        {
            intersection.flag = false;
            return intersection;
        }

        intersection.flag = true;
        double t1 = -sqrt(delta)-b;
        double t2 = sqrt(delta)-b;
        if (t2 < 0) intersection.flag = false;
        else {
            intersection.flag = true;
            intersection.t = t1 >= 0 ? t1 : t2;
        }
        intersection.P = ray.origin + ray.direction * intersection.t;
        Vector CP = intersection.P - C;
        CP.normalization();
        intersection.N = CP;
        if (hallow==true){
             intersection.N = (-1) * intersection.N;
        }
        return intersection;
    }
};

class BBox 
{
public:
    Vector min_box = Vector(0, 0, 0);
    Vector max_box = Vector(0, 0, 0);
    BBox(){}
    BBox(Vector &min_box, Vector &max_box)
    {
        min_box = min_box;
        max_box = max_box;
    }

    bool intersect(Ray &ray, double &int_dist)
    {
		Vector x = (min_box-ray.origin) / ray.direction;
		Vector y = (max_box-ray.origin) / ray.direction;
		Vector t0, t1;
		for (int i = 0; i < 3; i++)
		{
			t1[i] = std::max(x[i], y[i]);
			t0[i] = std::min(x[i], y[i]);
		}
		double maxi = std::max(t0[0], std::max(t0[1], t0[2]));
        double mini = std::min(t1[0], std::min(t1[1], t1[2]));
        if (maxi > 0&& mini > maxi)
        {
            int_dist = maxi;
            return true;
        }
        return false;
    }
};

class Node
{
public:
    int start;
    int end;
    BBox bbox;
    Node *left_child;
    Node *right_child;
};

class TriangleIndices {
public:

    TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
    };
    int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
    int uvi, uvj, uvk;  // indices within the uv coordinates array
    int ni, nj, nk;  // indices within the normals array
    int group;       // face group
};
 
 
class TriangleMesh : public Geometry
{
public:
	~TriangleMesh() {}
    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;

    double scaling;
    Vector translation;
    Node start;

	TriangleMesh(Vector albedo_val, double scaling_val, Vector translation_val)
    {
        albedo = albedo_val;
        scaling = scaling_val;
        translation = translation_val;
    }
    BBox compute_bbox(int begin, int end)
	{
		BBox bbox;

		bbox.max_box = Vector(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());
		bbox.min_box = Vector(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
		for (int i = begin; i < end; i++)
        {
        for (int j = 0; j < 3; j++)
        {
            bbox.min_box[j] = std::min(vertices[indices[i].vtxi][j],bbox.min_box[j]);
            bbox.max_box[j] = std::max(vertices[indices[i].vtxi][j], bbox.max_box[j]);
            bbox.min_box[j] = std::min(vertices[indices[i].vtxj][j], bbox.min_box[j]);
            bbox.max_box[j] = std::max(vertices[indices[i].vtxj][j], bbox.max_box[j]);
            bbox.min_box[j] = std::min(vertices[indices[i].vtxk][j],bbox.min_box[j]);
            bbox.max_box[j] = std::max(vertices[indices[i].vtxk][j],bbox.max_box[j]);
        }
        }
		return bbox;
	}

	void readOBJ(const char *obj)
	{
		char matfile[255];
		char grp[255];

		FILE *f;
		f = fopen(obj, "r");
		int curGroup = -1;
		while (!feof(f))
		{
			char line[255];
			if (!fgets(line, 255, f))
				break;

			std::string linetrim(line);
			linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
			strcpy(line, linetrim.c_str());

			if (line[0] == 'u' && line[1] == 's')
			{
				sscanf(line, "usemtl %[^\n]\n", grp);
				curGroup++;
			}

			if (line[0] == 'v' && line[1] == ' ')
			{
				Vector vec;

				Vector col;
				if (sscanf(line, "v %lf %lf %lf %lf %lf %lf\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6)
				{
					col[0] = std::min(1., std::max(0., col[0]));
					col[1] = std::min(1., std::max(0., col[1]));
					col[2] = std::min(1., std::max(0., col[2]));

					vertices.push_back(vec);
					vertexcolors.push_back(col);
				}
				else
				{
					sscanf(line, "v %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
					vertices.push_back(vec);
				}
			}
			if (line[0] == 'v' && line[1] == 'n')
			{
				Vector vec;
				sscanf(line, "vn %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
				normals.push_back(vec);
			}
			if (line[0] == 'v' && line[1] == 't')
			{
				Vector vec;
				sscanf(line, "vt %lf %lf\n", &vec[0], &vec[1]);
				uvs.push_back(vec);
			}
			if (line[0] == 'f')
			{
				TriangleIndices t;
				int i0, i1, i2, i3;
				int j0, j1, j2, j3;
				int k0, k1, k2, k3;
				int nn;
				t.group = curGroup;

				char *consumedline = line + 1;
				int offset;

				nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
				if (nn == 9)
				{
					if (i0 < 0)
						t.vtxi = vertices.size() + i0;
					else
						t.vtxi = i0 - 1;
					if (i1 < 0)
						t.vtxj = vertices.size() + i1;
					else
						t.vtxj = i1 - 1;
					if (i2 < 0)
						t.vtxk = vertices.size() + i2;
					else
						t.vtxk = i2 - 1;
					if (j0 < 0)
						t.uvi = uvs.size() + j0;
					else
						t.uvi = j0 - 1;
					if (j1 < 0)
						t.uvj = uvs.size() + j1;
					else
						t.uvj = j1 - 1;
					if (j2 < 0)
						t.uvk = uvs.size() + j2;
					else
						t.uvk = j2 - 1;
					if (k0 < 0)
						t.ni = normals.size() + k0;
					else
						t.ni = k0 - 1;
					if (k1 < 0)
						t.nj = normals.size() + k1;
					else
						t.nj = k1 - 1;
					if (k2 < 0)
						t.nk = normals.size() + k2;
					else
						t.nk = k2 - 1;
					indices.push_back(t);
				}
				else
				{
					nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
					if (nn == 6)
					{
						if (i0 < 0)
							t.vtxi = vertices.size() + i0;
						else
							t.vtxi = i0 - 1;
						if (i1 < 0)
							t.vtxj = vertices.size() + i1;
						else
							t.vtxj = i1 - 1;
						if (i2 < 0)
							t.vtxk = vertices.size() + i2;
						else
							t.vtxk = i2 - 1;
						if (j0 < 0)
							t.uvi = uvs.size() + j0;
						else
							t.uvi = j0 - 1;
						if (j1 < 0)
							t.uvj = uvs.size() + j1;
						else
							t.uvj = j1 - 1;
						if (j2 < 0)
							t.uvk = uvs.size() + j2;
						else
							t.uvk = j2 - 1;
						indices.push_back(t);
					}
					else
					{
						nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
						if (nn == 3)
						{
							if (i0 < 0)
								t.vtxi = vertices.size() + i0;
							else
								t.vtxi = i0 - 1;
							if (i1 < 0)
								t.vtxj = vertices.size() + i1;
							else
								t.vtxj = i1 - 1;
							if (i2 < 0)
								t.vtxk = vertices.size() + i2;
							else
								t.vtxk = i2 - 1;
							indices.push_back(t);
						}
						else
						{
							nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
							if (i0 < 0)
								t.vtxi = vertices.size() + i0;
							else
								t.vtxi = i0 - 1;
							if (i1 < 0)
								t.vtxj = vertices.size() + i1;
							else
								t.vtxj = i1 - 1;
							if (i2 < 0)
								t.vtxk = vertices.size() + i2;
							else
								t.vtxk = i2 - 1;
							if (k0 < 0)
								t.ni = normals.size() + k0;
							else
								t.ni = k0 - 1;
							if (k1 < 0)
								t.nj = normals.size() + k1;
							else
								t.nj = k1 - 1;
							if (k2 < 0)
								t.nk = normals.size() + k2;
							else
								t.nk = k2 - 1;
							indices.push_back(t);
						}
					}
				}

				consumedline = consumedline + offset;

				while (true)
				{
					if (consumedline[0] == '\n')
						break;
					if (consumedline[0] == '\0')
						break;
					nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
					TriangleIndices t2;
					t2.group = curGroup;
					if (nn == 3)
					{
						if (i0 < 0)
							t2.vtxi = vertices.size() + i0;
						else
							t2.vtxi = i0 - 1;
						if (i2 < 0)
							t2.vtxj = vertices.size() + i2;
						else
							t2.vtxj = i2 - 1;
						if (i3 < 0)
							t2.vtxk = vertices.size() + i3;
						else
							t2.vtxk = i3 - 1;
						if (j0 < 0)
							t2.uvi = uvs.size() + j0;
						else
							t2.uvi = j0 - 1;
						if (j2 < 0)
							t2.uvj = uvs.size() + j2;
						else
							t2.uvj = j2 - 1;
						if (j3 < 0)
							t2.uvk = uvs.size() + j3;
						else
							t2.uvk = j3 - 1;
						if (k0 < 0)
							t2.ni = normals.size() + k0;
						else
							t2.ni = k0 - 1;
						if (k2 < 0)
							t2.nj = normals.size() + k2;
						else
							t2.nj = k2 - 1;
						if (k3 < 0)
							t2.nk = normals.size() + k3;
						else
							t2.nk = k3 - 1;
						indices.push_back(t2);
						consumedline = consumedline + offset;
						i2 = i3;
						j2 = j3;
						k2 = k3;
					}
					else
					{
						nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
						if (nn == 2)
						{
							if (i0 < 0)
								t2.vtxi = vertices.size() + i0;
							else
								t2.vtxi = i0 - 1;
							if (i2 < 0)
								t2.vtxj = vertices.size() + i2;
							else
								t2.vtxj = i2 - 1;
							if (i3 < 0)
								t2.vtxk = vertices.size() + i3;
							else
								t2.vtxk = i3 - 1;
							if (j0 < 0)
								t2.uvi = uvs.size() + j0;
							else
								t2.uvi = j0 - 1;
							if (j2 < 0)
								t2.uvj = uvs.size() + j2;
							else
								t2.uvj = j2 - 1;
							if (j3 < 0)
								t2.uvk = uvs.size() + j3;
							else
								t2.uvk = j3 - 1;
							consumedline = consumedline + offset;
							i2 = i3;
							j2 = j3;
							indices.push_back(t2);
						}
						else
						{
							nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
							if (nn == 2)
							{
								if (i0 < 0)
									t2.vtxi = vertices.size() + i0;
								else
									t2.vtxi = i0 - 1;
								if (i2 < 0)
									t2.vtxj = vertices.size() + i2;
								else
									t2.vtxj = i2 - 1;
								if (i3 < 0)
									t2.vtxk = vertices.size() + i3;
								else
									t2.vtxk = i3 - 1;
								if (k0 < 0)
									t2.ni = normals.size() + k0;
								else
									t2.ni = k0 - 1;
								if (k2 < 0)
									t2.nj = normals.size() + k2;
								else
									t2.nj = k2 - 1;
								if (k3 < 0)
									t2.nk = normals.size() + k3;
								else
									t2.nk = k3 - 1;
								consumedline = consumedline + offset;
								i2 = i3;
								k2 = k3;
								indices.push_back(t2);
							}
							else
							{
								nn = sscanf(consumedline, "%u%n", &i3, &offset);
								if (nn == 1)
								{
									if (i0 < 0)
										t2.vtxi = vertices.size() + i0;
									else
										t2.vtxi = i0 - 1;
									if (i2 < 0)
										t2.vtxj = vertices.size() + i2;
									else
										t2.vtxj = i2 - 1;
									if (i3 < 0)
										t2.vtxk = vertices.size() + i3;
									else
										t2.vtxk = i3 - 1;
									consumedline = consumedline + offset;
									i2 = i3;
									indices.push_back(t2);
								}
								else
								{
									consumedline = consumedline + 1;
								}
							}
						}
					}
				}
			}
		}
		fclose(f);
	}
 void BVH(Node *node, int start, int end)
    {

        BBox bbox = compute_bbox(start, end);
        node->bbox = bbox;
        node->start = start;
        node->end = end;
		node->left_child = nullptr;
		node->right_child = nullptr;

        Vector diagonal = node->bbox.max_box - node->bbox.min_box;
        Vector middle_diagonal = node->bbox.min_box + diagonal * 0.5;
        int path;
        if ( std::max(diagonal[0], diagonal[2]) <diagonal[1])
		{
            path = 1;
        }
        else if ( std::max(diagonal[1], diagonal[2])< diagonal[0])
        {
			path = 0;
        }
        else
		{	
            path = 2;
        }

        int pivot = start;
        for (int i = start; i < end; i++)
        {
            Vector barycenter = (vertices[indices[i].vtxi] + vertices[indices[i].vtxj] + vertices[indices[i].vtxk]) / 3;

            if (barycenter[path] < middle_diagonal[path])
            {
                std::swap(indices[i], indices[pivot]);
                pivot++;
            }
        }

        if (end-pivot <= 5|| pivot-start <= 5||end-start < 10)
        {
            return;
        }
        node->left_child = new Node();
        node->right_child = new Node();
        BVH(node->left_child, start, pivot);
        BVH(node->right_child, pivot, end);
    }

Intersection intersect(Ray &ray)
    {
        Intersection ray_intersection;
        double int_dist;
        if (!start.bbox.intersect(ray, int_dist))
        {
            return ray_intersection;
        }

        std::list<Node *> Nodelist;
        Nodelist.push_back(&start);
        double t = std::numeric_limits<double>::max();
        while (!Nodelist.empty())
        {
            Node *NodeTemp = Nodelist.back();
            Nodelist.pop_back();

            double t_left_child;
			double t_right_child;
            if (NodeTemp->left_child == nullptr && NodeTemp->right_child == nullptr)
            {
                for (int i = NodeTemp->start; i < NodeTemp->end; i++)
                {
                    TriangleIndices triangle = indices[i];

                    Vector A = vertices[triangle.vtxi];
                    Vector B = vertices[triangle.vtxj];
                    Vector C = vertices[triangle.vtxk];
                    Vector e1 = B - A;
                    Vector e2 = C - A;
                    Vector N = cross(e1, e2);

                    double res_t = dot(A - ray.origin, N) / dot(ray.direction, N);

                    if (0 < res_t && res_t < t)
                    {
                        double gamma = -dot(e1, cross(A - ray.origin, ray.direction))/ dot(ray.direction, N);
                        double beta = dot(e2, cross(A - ray.origin, ray.direction)) / dot(ray.direction, N);
                        double alpha = 1 - beta - gamma;

                        if (alpha > 0 && gamma > 0 &&  beta > 0)
                        {
                            t = res_t;
                            ray_intersection.t = t;
                            N.normalization();
                            ray_intersection.N = N;
                            ray_intersection.P = A + e1 * beta + e2 * gamma;
                            ray_intersection.flag = true;
                        }

                    }
                }
            }

            if (NodeTemp->left_child != nullptr && NodeTemp->left_child->bbox.intersect(ray, t_left_child))
			{
				if (t_left_child < t)
                {
					Nodelist.push_back(NodeTemp->left_child);
                }
            }
            if (NodeTemp->right_child != nullptr && NodeTemp->right_child->bbox.intersect(ray, t_right_child))
			{
				if (t_right_child < t)
				{
                    Nodelist.push_back(NodeTemp->right_child);
                }
			}
        }
        return ray_intersection;
    }

};

// Box-Muller transform
void boxMuller(double stdev, double &x, double &y){
    double r1 = uniform(engine);
    double r2 = uniform(engine);
    x = sqrt(-2 * log(r1)) * cos(2*M_PI*r2)*stdev;
    y = sqrt(-2 * log(r1)) * sin(2*M_PI*r2)*stdev;
}





//Scene class
class Scene
{
    public:
    std::vector<Geometry*> objects;
    Lighting light;
    
    Scene(Lighting lightnew, int object_type)
    {
        light = lightnew;
        // Scene
        Sphere *left_child = new Sphere(Vector(-1000, 0, 0), Vector(0, 1, 1), 940);
        Sphere *right_child = new Sphere(Vector(1000, 0, 0), Vector(1, 1, 0), 940);
        Sphere *top = new Sphere(Vector(0, 1000, 0), Vector(1, 0, 0), 940);
        Sphere *bottom = new Sphere(Vector(0, -1000, 0), Vector(0, 0, 1), 990);
        Sphere *front = new Sphere(Vector(0, 0, -1000), Vector(0, 1, 0), 940);
        Sphere *back = new Sphere(Vector(0, 0, 1000), Vector(1,0,1), 940);
        
        objects.push_back(left_child);
        objects.push_back(right_child);
        objects.push_back(top);
        objects.push_back(bottom);
        objects.push_back(front);
        objects.push_back(back);
            
        if(object_type == 1)
        {
            TriangleMesh *mesh = new TriangleMesh(Vector(1, 1, 1), 0.6, Vector(0, -10, 0));
            mesh->readOBJ("cat.obj");
            for (int i = 0; i < mesh->vertices.size(); i++)
            {
                 mesh->vertices[i] = mesh->vertices[i]*0.6+Vector(0, -10, 0);
            }
            mesh->BVH(&mesh->start, 0, mesh->indices.size());
            objects.push_back(mesh);
        }
        if(object_type == 2)
        {
            Sphere *mirror_sph = new Sphere(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), 10, false, true);
            objects.push_back(mirror_sph);
        }
        if(object_type == 3)
        {
            Sphere *center_sph = new Sphere(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), 10, false, true);
            Sphere *center_sph_hallow = new Sphere(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), 9, false, true,true);
            objects.push_back(center_sph);
            objects.push_back(center_sph_hallow);
        }
        if(object_type == 4)
        {
            Sphere *transparent_sph = new Sphere(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), 10, false, true);
            objects.push_back(transparent_sph);
        }

        if(object_type == 5)
        {

            Sphere *right_child_sph = new Sphere(Vector(20, 0, 0), Vector(0.5, 0.5, 0.5), 10, false, true);
            Sphere *center_sph_hallow = new Sphere(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), 9, false, true,true);
            Sphere *center_sph = new Sphere(Vector(0, 0, 0), Vector(0.5, 0.5, 0.5), 10, false, true);
            Sphere *left_child_sph = new Sphere(Vector(-20, 0, 0), Vector(0.5, 0.5, 0.5), 10, true, false);
            objects.push_back(left_child_sph);
            objects.push_back(center_sph);
            objects.push_back(right_child_sph);
            objects.push_back(center_sph_hallow);
        }


    }
    

    Vector getColor(Ray &ray, int depth, bool indirect_light, bool fresnel)
    {
        Intersection ray_intersect = intersect(ray);

        if (!(depth > 0 && ray_intersect.flag))
        {
            return Vector(0, 0, 0);
        }

        if (objects[ray_intersect.id]->transparent)
        {
            double n1 = 1;
            double n2 = 1.4;
            Vector u = ray.direction - 2 * dot(ray.direction, ray_intersect.N) * ray_intersect.N;
            Vector O = ray_intersect.P + 0.001 * ray_intersect.N;
            
            Ray reflect(O, u);
            Vector Normal = ray_intersect.N;
            if (dot(ray.direction, Normal) > 0)
            {
                std::swap(n1, n2);
                Normal = -1 * Normal;
            }
            if ((1 - pow((n1 / n2), 2) * (1 - pow(dot(ray.direction, Normal), 2))) < 0)
            {
                return getColor(reflect, depth - 1,indirect_light, fresnel);
            }
            Vector wt =  (ray.direction - dot(ray.direction, Normal) * Normal) * (n1 / n2);

            Vector wn = -sqrt((1 - pow((n1 / n2), 2) * (1 - pow(dot(ray.direction, Normal), 2)))) * Normal;

            Ray refract(ray_intersect.P + 0.001 * wt + wn, wt + wn);

            if (fresnel == true)
            {
				double k0 = std::pow((n1 - n2) / (n1 + n2), 2);
				double R = k0 + (1 - k0) * std::pow((1 - std::abs(dot(Normal, ray.direction))), 5);

				if (uniform(engine) < R) return getColor(reflect, depth - 1,indirect_light, fresnel);
            }
            return getColor(refract, depth - 1,indirect_light, fresnel);
        }

        if (objects[ray_intersect.id]->mirror)
        {  
            Vector u = ray.direction - 2 * ray_intersect.N * dot(ray.direction, ray_intersect.N);
            Vector O = ray_intersect.P + 0.001 * ray_intersect.N;
            Ray reflect(O , u);
            return getColor(reflect, depth - 1,indirect_light, fresnel);
        }

        Vector u_intersect = light.position - ray_intersect.P;
        double norm = u_intersect.norm();
        u_intersect.normalization();

        Ray light_ray(ray_intersect.P + 0.01 * ray_intersect.N, u_intersect);

        double shadow = 1;
        if (intersect(light_ray).flag && (intersect(light_ray).t * intersect(light_ray).t < norm))
        {
            shadow = 0;
        }

        Vector color = shadow * light.intensity * objects[ray_intersect.id]->albedo * std::max(0., dot(u_intersect, ray_intersect.N)) / (4 * M_PI * M_PI * norm * norm);

        if(indirect_light == true)
        {
            Ray random_ray(ray_intersect.P, random_cos(ray_intersect.N));
            color = color + objects[ray_intersect.id]->albedo * getColor(random_ray, depth - 1, indirect_light, fresnel);
        }

        return color;
    } 

    Intersection intersect(Ray &ray)
    {
        Intersection intersection;
        for (int i = 0; i < objects.size(); i++)
        {
            Intersection ray_intersect = objects[i]->intersect(ray); 
            if (ray_intersect.flag && (!intersection.flag||ray_intersect.t < intersection.t))
            {
                ray_intersect.id = i;
                intersection = ray_intersect;
            }
        }
        return intersection;
    }
}; 

// Main function

int main()
{
    int object_type = 1; // 1 for cat, 2 for mirror sphere, 3 for hallow sphere, 4 for transparent sphere, 5 for all 3 spheres
    int W = 512;
    int H = 512;
    
    Lighting lighting_scene;
    bool fresnel = false;
    bool indirect_lightning = false;
    Vector Origin = Vector(0, 0, 55);
    Vector L = Vector(-15, 30, 45);
	double I = 2e10;
    
    lighting_scene.position = L;
    lighting_scene.intensity = I;
    int paths_per_pixel = 64;
    Scene scene(lighting_scene, object_type);
    std::vector<unsigned char> image(W * H * 3, 0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            Vector color(0, 0, 0);
            
            for (int k = 0; k < paths_per_pixel; k++) 
            {
                double x, y;
                boxMuller(1, x, y);
                Vector direction((y + j)- W/ 2 + 0.5, H/2-(x + i) - 0.5, -W /(2 * tan(M_PI / 6)));
                direction.normalization();
                Ray ray(Origin, direction);
                color = color + scene.getColor(ray, 5, indirect_lightning, fresnel) / paths_per_pixel;

			image[(i * W + j) * 3 + 0] = std::min(255.0, std::pow(color[0], 1 / 2.2));
			image[(i * W + j) * 3 + 1] = std::min(255.0, std::pow(color[1], 1 / 2.2));
			image[(i * W + j) * 3 + 2] = std::min(255.0, std::pow(color[2], 1 / 2.2));
        }
    }
    }
    stbi_write_png("image.png", W, H, 3, &image[0], 0);

    return 0;
}
