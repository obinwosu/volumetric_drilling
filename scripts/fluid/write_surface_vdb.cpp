// Write or inspect a Maelstrom "surface" FloatGrid. fileToVDB only accepts
// a grid with that name. This is a host tool; it does not touch AMBF or RTX.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/Interpolation.h>

namespace {

constexpr char kMagic[8] = {'F', 'I', 'V', 'R', 'S', 'F', '3', '2'};

struct DenseHeader {
  char magic[8];
  std::int32_t nx, ny, nz;
  double origin[3];
  double spacing;
};

void check(std::istream &in, const char *what) {
  if (!in) {
    throw std::runtime_error(std::string("failed reading ") + what);
  }
}

template <typename T>
T read_pod(std::istream &in, const char *what) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  check(in, what);
  return value;
}

DenseHeader read_header(std::istream &in) {
  DenseHeader header{};
  in.read(header.magic, 8);
  check(in, "dense magic");
  if (std::memcmp(header.magic, kMagic, 8) != 0) {
    throw std::runtime_error("dense file magic is not FIVRSF32");
  }
  header.nx = read_pod<std::int32_t>(in, "nx");
  header.ny = read_pod<std::int32_t>(in, "ny");
  header.nz = read_pod<std::int32_t>(in, "nz");
  header.origin[0] = read_pod<double>(in, "origin.x");
  header.origin[1] = read_pod<double>(in, "origin.y");
  header.origin[2] = read_pod<double>(in, "origin.z");
  header.spacing = read_pod<double>(in, "spacing");
  if (header.nx <= 0 || header.ny <= 0 || header.nz <= 0 || header.spacing <= 0.0) {
    throw std::runtime_error("dense header has non-positive size or spacing");
  }
  return header;
}

void write_vdb(const std::string &dense_path, const std::string &vdb_path) {
  std::ifstream in(dense_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open dense file " + dense_path);
  }
  const DenseHeader header = read_header(in);
  const std::size_t count =
      static_cast<std::size_t>(header.nx) * static_cast<std::size_t>(header.ny) *
      static_cast<std::size_t>(header.nz);
  std::vector<float> values(count);
  in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  check(in, "dense voxels");

  openvdb::initialize();
  openvdb::math::Transform::Ptr xform =
      openvdb::math::Transform::createLinearTransform(header.spacing);
  xform->postTranslate(openvdb::Vec3d(header.origin[0], header.origin[1], header.origin[2]));

  float background = 0.0f;
  for (float value : values) {
    background = std::max(background, value);
  }
  if (background <= 0.0f) {
    background = static_cast<float>(header.spacing);
  }

  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(background);
  grid->setName("surface");
  grid->setGridClass(openvdb::GRID_LEVEL_SET);
  grid->setTransform(xform);

  const openvdb::CoordBBox bbox(
      openvdb::Coord(0, 0, 0),
      openvdb::Coord(header.nx - 1, header.ny - 1, header.nz - 1));
  openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(bbox, values.data());
  openvdb::tools::copyFromDense(dense, *grid, /*tolerance=*/0.0f);

  openvdb::io::File file(vdb_path);
  openvdb::GridPtrVec grids;
  grids.push_back(grid);
  file.write(grids);
  file.close();
}

openvdb::FloatGrid::Ptr load_surface(const std::string &vdb_path) {
  openvdb::initialize();
  openvdb::io::File file(vdb_path);
  file.open();
  openvdb::GridBase::Ptr base;
  for (auto name = file.beginName(); name != file.endName(); ++name) {
    if (name.gridName() == "surface") {
      base = file.readGrid(name.gridName());
      break;
    }
  }
  file.close();
  if (!base) {
    throw std::runtime_error("no grid named surface");
  }
  openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
  if (!grid) {
    throw std::runtime_error("surface grid is not FloatGrid");
  }
  return grid;
}

void sample_vdb(const std::string &vdb_path, double x, double y, double z) {
  openvdb::FloatGrid::Ptr grid = load_surface(vdb_path);
  openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid);
  std::cout << sampler.wsSample(openvdb::Vec3d(x, y, z)) << "\n";
}

void inspect_vdb(const std::string &vdb_path) {
  openvdb::FloatGrid::Ptr grid = load_surface(vdb_path);
  const openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
  const openvdb::Vec3d vmin = grid->indexToWorld(box.getStart());
  const openvdb::Vec3d vmax = grid->indexToWorld(box.getEnd());
  const openvdb::Vec3d voxel = grid->voxelSize();
  const openvdb::Coord dim = box.dim();
  openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid);
  const openvdb::Vec3d mid = 0.5 * (vmin + vmax);
  std::cout << "name " << grid->getName() << "\n";
  std::cout << "class " << grid->getGridClass() << "\n";
  std::cout << "active " << grid->activeVoxelCount() << "\n";
  std::cout << "dim " << dim.x() << " " << dim.y() << " " << dim.z() << "\n";
  std::cout << "origin " << vmin.x() << " " << vmin.y() << " " << vmin.z() << "\n";
  std::cout << "corner " << vmax.x() << " " << vmax.y() << " " << vmax.z() << "\n";
  std::cout << "spacing " << voxel.x() << " " << voxel.y() << " " << voxel.z() << "\n";
  std::cout << "mid " << sampler.wsSample(mid) << "\n";
  std::cout << "level_set " << (grid->getGridClass() == openvdb::GRID_LEVEL_SET) << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "write") {
      write_vdb(argv[2], argv[3]);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "inspect") {
      inspect_vdb(argv[2]);
      return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "sample") {
      sample_vdb(argv[2], std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5]));
      return 0;
    }
    std::cerr << "usage: write_surface_vdb write <dense.bin> <out.vdb>\n"
              << "       write_surface_vdb inspect <in.vdb>\n"
              << "       write_surface_vdb sample <in.vdb> <x> <y> <z>\n";
    return 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
