import vtk
from vtkmodules.util import numpy_support
import xml.etree.ElementTree as ET
import numpy as np
# from tqdm import tqdm
import json

# zero-pad an axis to length
def np_zeropad(arr, length, axis):
  pad_shape = list(arr.shape)
  pad_shape[axis] = length - pad_shape[axis]
  npad = np.zeros(pad_shape)
  padded = np.concatenate((arr, npad), axis=axis)
  return padded

# create a mesh matrix of shape (D1, ..., Di, #ofDim). Di = dimension i length
def get_mesh(*dims):
  mesh_coords = []
  mesh_shape = np.array([len(dim) for dim in dims])
  for i, dim in enumerate(dims):
    # expand shape to everywhere 1 except for the dimension index
    dim_shape = np.ones(len(dims), dtype=int)
    dim_shape[i] = len(dim)
    dim_coords = dim.reshape(dim_shape)

    # repeat the length 1 dimension to match the other dimension lengths
    dim_repeats = mesh_shape.copy()
    dim_repeats[i] = 1
    dim_coords = np.tile(dim_coords, dim_repeats)
    # print(mesh_shape, hex(id(mesh_shape)), hex(id(dim_repeats)))
    mesh_coords.append(dim_coords[..., None])
  
  mesh_coords = np.concatenate(mesh_coords, axis=-1)
  print("meshe generated:", mesh_coords.shape)
  return mesh_coords

def get_vtu(position:np.array, scalar_fields={}, vector_fields={}):
  vtk_position = numpy_support.numpy_to_vtk(position)
  points = vtk.vtkPoints()
  points.SetData(vtk_position)
  data_save = vtk.vtkUnstructuredGrid()
  data_save.SetPoints(points)

  # setup values
  pd = data_save.GetPointData()
  for i, (k, v) in enumerate(scalar_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetScalars(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  for i, (k, v) in enumerate(vector_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetVectors(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  return data_save

def write_vtu(fpath:str, vtu):
  writer = vtk.vtkXMLDataSetWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vtu)
  writer.Write()

def get_vts(dims, points, scalar_fields={}, vector_fields={}):
  vtk_grid = vtk.vtkStructuredGrid()
  vtk_grid.SetDimensions(dims)

  # setup grid
  point_coords = numpy_support.numpy_to_vtk(points.reshape(-1, 3))
  vtkPoints = vtk.vtkPoints()
  vtkPoints.SetData(point_coords)
  vtk_grid.SetPoints(vtkPoints)

  # setup values
  pd = vtk_grid.GetPointData()
  for i, (k, v) in enumerate(scalar_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetScalars(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  for i, (k, v) in enumerate(vector_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetVectors(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  return vtk_grid

def write_vts(fpath, vts):
  writer = vtk.vtkXMLStructuredGridWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vts)
  writer.Write()


# x,y,z coordiantes either numpy / vtk array 
def get_vtr(dims, xCoords, yCoords, zCoords, scalar_fields={}, vector_fields={}):
  assert type(xCoords) == type(yCoords) and type(yCoords) == type(zCoords)
  assert isinstance(xCoords, np.ndarray) or isinstance(xCoords, vtk.vtkDataArray)

  grid = vtk.vtkRectilinearGrid()
  grid.SetDimensions(dims)

  if isinstance(xCoords, np.ndarray):
    xCoords = numpy_support.numpy_to_vtk(xCoords)
    yCoords = numpy_support.numpy_to_vtk(yCoords)
    zCoords = numpy_support.numpy_to_vtk(zCoords)

  grid.SetXCoordinates(xCoords)
  grid.SetYCoordinates(yCoords)
  grid.SetZCoordinates(zCoords)

  pd = grid.GetPointData()
  for i, (k, v) in enumerate(scalar_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetScalars(vtk_array)
    else:
      pd.AddArray(vtk_array)
  
  for i, (k, v) in enumerate(vector_fields.items()):
    vtk_array = numpy_support.numpy_to_vtk(v)
    vtk_array.SetName(k)
    if i == 0:
      pd.SetVectors(vtk_array)
    else:
      pd.AddArray(vtk_array)

  return grid

def write_vtr(fpath, vtr):
  writer = vtk.vtkXMLRectilinearGridWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vtr)
  writer.Write()

# read a .raw file to vti
#  bbox: (2, numdim), e.g. [[xmin,ymin], [xmax, ymax]]
def read_raw_vti(fpath, arr_name, bbox):
  reader = vtk.vtkImageReader()
  reader.SetFileName(fpath)
  reader.SetScalarArrayName(arr_name)
  reader.SetDataByteOrderToLittleEndian()
  reader.SetDataScalarTypeToFloat()
  reader.SetDataExtent(bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1], bbox[0][2], bbox[1][2])
  reader.SetFileDimensionality(3)
  reader.SetDataOrigin(0, 0, 0)
  reader.SetDataSpacing(1, 1, 1)
  reader.Update()
  return reader.GetOutput()

def write_vti(fpath, vti):
  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vti)
  writer.Write()

def get_vtp_polyline(sl_coords: np.array, sllens: np.array):
  vtp = vtk.vtkPolyData()
  points = vtk.vtkPoints()
  points.SetData(numpy_support.numpy_to_vtk(sl_coords))
  vtp.SetPoints(points)
  vtkCells = vtk.vtkCellArray()

  seedids = [0]*len(sllens)
  for i, sllen in enumerate(sllens):
    id_np = i*np.ones((sllen,1))
    seedids[i] = id_np
  seedids = np.concatenate(seedids)

  vtk_seedids = numpy_support.numpy_to_vtk(seedids)
  vtk_seedids.SetName("seedID")
  vtp.GetPointData().AddArray(vtk_seedids)

  curr_len = 0
  for sllen in sllens:
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(sllen)
    for i in range(sllen):
      polyLine.GetPointIds().SetId(i, curr_len+i)
    curr_len += sllen
    vtkCells.InsertNextCell(polyLine)
  vtp.SetLines(vtkCells)
  return vtp

def write_vtp(fpath, vtp):
  writer = vtk.vtkXMLPolyDataWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vtp)
  writer.Write()

# return vtkFloatArray holding data like np.arange(args)
def vtk_arange(*args):
  arr = vtk.vtkFloatArray()
  coords = np.arange(*args)
  arr.Allocate(len(coords))
  for i in coords:
    arr.InsertNextValue(i)
  return arr

# return vtkFloatArray holding data like np.linspace(args)
def vtk_linspace(*args):
  arr = vtk.vtkFloatArray()
  coords = np.linspace(*args)
  arr.Allocate(len(coords))
  for i in coords:
    arr.InsertNextValue(i)
  return arr

def parse_paraview_tf(filename: str):
  # parsing transfer function exported from Paraview
  with open(filename, "r") as f:
    tf_json = json.load(f)
  tf_json = tf_json[0]
  opacityPoints = np.array(tf_json['Points']).reshape(-1, 4)
  opacityPoints = opacityPoints[:, :2]
  colorPoints = np.array(tf_json['RGBPoints']).reshape(-1, 4)
  return opacityPoints, colorPoints

def vtkMatToNumpy(vtkMat):
  if isinstance(vtkMat, vtk.vtkMatrix4x4):
    matrixSize = 4
  elif isinstance(vtkMat, vtk.vtkMatrix3x3):
    matrixSize = 3
  else:
    raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
  narray = np.eye(matrixSize)
  vtkMat.DeepCopy(narray.ravel(), vtkMat)
  return narray.copy()

def write_vtm(fpath, vtm):
  writer = vtk.vtkXMLMultiBlockDataWriter()
  writer.SetFileName(fpath)
  writer.SetInputData(vtm)
  writer.Write()

def write_pvd(vtkPath:list, outPath:str, timesteps=None):
  # root tag
  vtkfile = ET.Element('VTKFile')
  vtkfile.set('type', 'Collection')
  
  # Collection is a child a VKTFile tag containing individual files as children
  collection = ET.SubElement(vtkfile, 'Collection')

  datasets = []
  for i, vtkp in enumerate(vtkPath):
    dataset = ET.SubElement(collection, 'DataSet')
    dataset.set('part', "0")
    # dataset.set('part', str(i))
    dataset.set('file', vtkp)
    datasets.append(dataset)
  
  # Enter timesteps if a list of timestep is provided
  if timesteps is not None:
    assert len(timesteps) == len(vtkPath)
    for dataset, ts in zip(datasets, timesteps):
      dataset.set('timestep', str(ts))
  
  pvd_et = ET.ElementTree(vtkfile)
  pvd_et.write(outPath, xml_declaration=True)