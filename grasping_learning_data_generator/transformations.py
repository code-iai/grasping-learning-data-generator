# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2009, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import math
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def get_quaternion_matrix(quaternion):
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)

    if nq < _EPS:
        return np.identity(4)

    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def calculate_object_faces(robot_to_object_transform):
    object_to_robot_transform = inverse_matrix(robot_to_object_transform)

    quaternion_matrix = get_quaternion_matrix(quaternion_from_matrix(object_to_robot_transform))
    robot_negative_x_vector = np.array([-quaternion_matrix[0][0], -quaternion_matrix[1][0], -quaternion_matrix[2][0]])
    robot_negative_z_vector = np.array([-quaternion_matrix[0][2], -quaternion_matrix[1][2], -quaternion_matrix[2][2]])

    return _calculate_vector_face(robot_negative_x_vector), _calculate_vector_face(robot_negative_z_vector)


def get_object_robot_transformation(robot_object_translation, robot_object_quaternion):
    transform_matrix = get_transform_matrix(robot_object_translation, robot_object_quaternion)
    object_to_robot_transform = inverse_matrix(transform_matrix)
    return object_to_robot_transform


def get_transform_matrix(translation, quaternion):
    translation_matrix = get_translation_matrix(translation)
    quaternion_matrix = get_quaternion_matrix(quaternion)
    transform_matrix = concatenate_matrices(translation_matrix, quaternion_matrix)

    return transform_matrix


def _calculate_vector_face(robot_vector):
    dimension = abs(robot_vector).argmax()
    value = robot_vector[dimension]

    if dimension == 0:
        if value > 0.:
            return ':FRONT'
        else:
            return ':BACK'
    elif dimension == 1:
        if value > 0.:
            return ':LEFT-SIDE'
        else:
            return ':RIGHT-SIDE'
    else:
        if value > 0.:
            return ':TOP'
        else:
            return ':BOTTOM'


def translation_from_matrix(matrix):
    return np.array(matrix, copy=False)[:3, 3].copy()


def get_translation_matrix(direction):
    matrix = np.identity(4)
    matrix[:3, 3] = direction[:3]
    return matrix


def concatenate_matrices(*matrices):
    matrix = np.identity(4)
    for i in matrices:
        matrix = np.dot(matrix, i)

    return matrix


def inverse_matrix(matrix):
    return np.linalg.inv(matrix)


def quaternion_from_matrix(matrix):
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q
