import numpy as np 
import meshio


from skspatial.objects import Plane
from skspatial.objects import Point
from skspatial.objects import Vector



class Levelset:
    """
    This class defines the level set function of a given geometrie mesh,
    we use a signed distance to define the levelset on the domain 
    :param file_name: Mesh file of the geometry, should be readable by Meshio, e.g stl, obj..etc
    """
    def __init__(self, file_mesh) -> None:

        self.mesh = meshio.read(file_mesh)
        self.points = self.getPoints()
        self.normals = self.getNormals()
        self.cells = self.getCells()

            
    def getNormals(self): 
        """
            Returns the normals of the mesh cells
        """
        return self.mesh.cell_data['facet_normals'][0].tolist()

    def getPoints(self):
        """
            Returns the nodes of the domain 
        """
        return self.mesh.points

    def getCells(self):
        """
            Returns the Cells of the domain 
        """
        return self.mesh.cells[0].data

    def _get_min_distance(self, target_point): 
        """
            Given a point in the space, this method returns the distance from the geometry
        """
        dist = np.linalg.norm(np.array(self.points[0]) - np.array(target_point))
        for pt in self.points: 
            tmp_dist = np.linalg.norm(np.array(pt) - np.array(target_point))
            if tmp_dist < dist: 
                dist = tmp_dist
        return dist

    def _get_cell_distance(self, cell_points, target_point): 
        """
            Given a point in the space and a cell of the mesh, this method returns the distance from the cell, it computes the mean distance with respect to the points of the cell
        """
        dist = 0
        for point in cell_points: 
            dist += np.linalg.norm(np.array(point) - np.array(target_point))
        dist /= 3
        return dist

    def _get_closest_cell(self, target_point):
        """
            Given a point in the space, this method returns the closest cell belonging to the mesh
        """
        cells = self.cells
        points = self.points
        cell_points = [points[cells[0][0]], points[cells[0][1]], points[cells[0][2]]]
        dist = self._get_cell_distance(cell_points, target_point)
        cell_index = 0
        for i, cell in enumerate(cells):
            cell_points = [points[cell[0]], points[cell[1]], points[cell[2]]]
            tmp_dist = self._get_cell_distance(cell_points, target_point)
            if tmp_dist < dist: 
                dist = tmp_dist
                cell_index = i
        return cell_index

        
    def _getProjection(self, point_of_cell, normal, target_point): 
        """
            Given a point in the space and cell of the domain, this method returns the orthogonal projection of the point on the cell
        """
        plane = Plane(point=point_of_cell, normal=normal)
        point = Point(target_point)
        point_projected = plane.project_point(point)
        return point_projected


    def _inside(self, target_point):
        """
            Given a point in the space, this method checks whether the point is located in or outside the mesh domain
        """
        points = self.mesh.points
        normals = self.normals
        cells = self.cells

        closest_cell_index = self._get_closest_cell(target_point)
        cell = cells[closest_cell_index]
        cell_points = [points[cell[0]], points[cell[1]], points[cell[2]]]
        normal = normals[closest_cell_index]

        target_point_proj = self._getProjection(cell_points[0], normal, target_point)
        vector_projection = Vector.from_points(target_point_proj, target_point)

        return normal@vector_projection < 0

    def phi(self, target_point): 
        """
            The main method of the class, it evaluates the levelSet function at a given point in the space
        """
        val_phi = self._get_min_distance(target_point)
        if self._inside(target_point): 
            val_phi = - val_phi
        return val_phi
    

if __name__ == "__main__":
    ls = Levelset("./data/refined_sphere_mesh.stl")

    print("This test is done with a Sphere with center [0.0, 0.0, 0.0] and radius of 0.5")

    print("############################################################")
    print("#  Inside points : Should give a negative levelset values  #")
    print("############################################################")

    print(ls.phi([-0.3, 0, .0]))
    print(ls.phi([0.3, 0, .0]))

    print(ls.phi([0, -0.3, .0]))
    print(ls.phi([0, 0.3, .0]))

    print(ls.phi([0, 0, -0.3]))
    print(ls.phi([0, 0, 0.3]))

    
    print("############################################################")
    print("#   Outside points : Should give a null levelset values    #")
    print("############################################################")


    print(ls.phi([-0.5, 0, .0]))
    print(ls.phi([0.5, 0, .0]))

    print(ls.phi([0, -0.5, .0]))
    print(ls.phi([0, 0.5, .0]))

    print(ls.phi([0, 0, -0.5]))
    print(ls.phi([0, 0, 0.5]))

    print("#############################################################")
    print("#  Outside points : Should give a positive levelset values  #")
    print("#############################################################")


    print(ls.phi([-1, 0, .0]))
    print(ls.phi([1, 0, .0]))

    print(ls.phi([0, -1, .0]))
    print(ls.phi([0, 1, .0]))

    print(ls.phi([0, 0, -1]))
    print(ls.phi([0, 0, 1]))

