#!/usr/bin/env python3

from numpy.core.fromnumeric import reshape
from evtk.hl import pointsToVTK
import numpy as np
import precice

def main():
    
    # number of nodes 

    n = 100
    h = 1 / n
    t = 0

    # generate mesh

    x = np.zeros(n+1)
    y = np.zeros(n+1)
    z = np.linspace(0,10,n+1)

    # initial data

    u = np.zeros((n+1,3))
    p = np.zeros(n+1)
    rhs = np.zeros(n+1)

    # preCICE setup

    participant_name = "Fluid1D"
    config_file_name = "../precice-config.xml"
    solver_process_index = 0
    solver_process_size = 1
    interface = precice.Interface(participant_name, config_file_name, solver_process_index, solver_process_size)

    mesh_name = "Fluid1D-Mesh"
    mesh_id = interface.get_mesh_id(mesh_name)

    velocity_name = "Velocity"
    velocity_id = interface.get_data_id(velocity_name, mesh_id)

    pressure_name = "Pressure"
    pressure_id = interface.get_data_id(pressure_name, mesh_id)

    positions = [[0, 0, z[-1]]]

    vertex_ids = interface.set_mesh_vertices(mesh_id, positions)

    precice_dt = interface.initialize()


    while interface.is_coupling_ongoing():
        if interface.is_action_required(
            precice.action_write_iteration_checkpoint()):

            u_iter = u
            p_iter = p
            F_iter = F

            interface.mark_action_fulfilled(
            precice.action_write_iteration_checkpoint())

        # determine time step size

        dt = 0.025
        dt = np.minimum(dt,precice_dt)

        # set boundary conditions

        u[0,2] = 1
        u[1,2] = 1  # dirichlet velocity inlet

        u[-1,2] = u[-2,2]   # neumann velocity outlet

        p[0] = p[1] # neumann pressure inlet
        if interface.is_read_data_available():  # get dirichlet outlet pressure from 3D solver
            p[n+1] = interface.read_scalar_data(pressure_id, vertex_ids)

        # compute right-hand side of 1D PPE

        for i in range(n):

            rhs[i+1] = 1 / dt * ((u[i+2,2] - u[i,2]) / 2*h)

        # solve the PPE using a SOR solver

        tolerance = 0.001
        error = 1
        omega = 1.8

        while error >= tolerance:

            p[0] = p[1] # renew neumann pressure inlet
            
            for i in range(n):
                p[i+1] = (1-omega) * p[i+1] + (((omega * h^2) / 2) * ((p[i] + p[i+2]) / h^2) - rhs[i+1])

            sum = 0
            for i in range(n):
                val = (p[i] - 2*p[i+1] + p[i+2]) / h^2 - rhs[i+1]
                sum += val*val

            error = np.sqrt(sum/n)

        # calculate new velocities

        for i in range(n):
            u[i+1,2] = u[i+1,2] - dt * ((p[i+2] - p[i+1]) / h)


        if interface.is_write_data_required(dt):    # write new velocities to 3D solver
            interface.write_vector_data(
            velocity_id, vertex_ids, u[-2])

        # transform data and write output data to vtk files

        u_print = np.reshape(u[:,2], n+1)
        p_print = np.reshape(p, n+1)

        u_print = np.ascontiguousarray(u_print, dtype=np.float32)
        p_print = np.ascontiguousarray(p_print, dtype=np.float32)

        filename = "./data/Fluid1D_" + str(t)
        pointsToVTK(filename, x, y, z, data = {"U" : u_print, "p" : p_print})

        # advance simulation time

        dt = interface.advance(dt)
        t = t + dt

        if interface.is_action_required(
            precice.action_read_iteration_checkpoint()):

            u = u_iter
            p = p_iter
            F = F_iter

            interface.mark_action_fulfilled(
            precice.action_read_iteration_checkpoint())

    interface.finalize()


if __name__ == "__main__":
    main()