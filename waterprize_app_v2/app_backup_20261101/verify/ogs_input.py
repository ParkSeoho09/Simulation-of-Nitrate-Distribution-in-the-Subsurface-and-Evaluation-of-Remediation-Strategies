"""OpenGeoSys 6 교차검증용 1D Richards 입력 생성기(입력 구조만 참고, OGS 코드 미포함).

참고한 입력 구조: Tests/Data/Parabolic/Richards/RichardsFlow_2d_small.prj (RICHARDS_FLOW, 곡선형 포화도·상대투수).
이 파일은 ogs 실행 파일이 있을 때만 쓰인다. 이 컨테이너에는 ogs가 없어 실행 검증을 하지 못했다(미검증).
"""
from __future__ import annotations

import os
import re

import numpy as np

from flow_richards import vg_kr, vg_Se
from params import SOIL_TEXTURES

RHO_G = 1000.0 * 9.81


def write_column_case(work, nz=200, L=1.0, h_top=-0.30, soil_name="사양토(Sandy loam)"):
    s = SOIL_TEXTURES[soil_name]
    mu = 1e-3
    k_int = s["Ks"] * mu / RHO_G                    # 고유투과도 m2
    x = np.linspace(0.0, L, nz + 1)                  # 1D 요소 축 = 표고
    # VTU (1D 선 요소, ASCII)
    pts = " ".join(f"{v:.9g} 0 0" for v in x)
    conn = " ".join(f"{i} {i + 1}" for i in range(nz))
    offs = " ".join(str(2 * (i + 1)) for i in range(nz))
    types = " ".join("3" for _ in range(nz))
    vtu = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="{nz + 1}" NumberOfCells="{nz}">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">{pts}</DataArray></Points>
<Cells><DataArray type="Int64" Name="connectivity" format="ascii">{conn}</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">{offs}</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">{types}</DataArray></Cells>
</Piece></UnstructuredGrid></VTKFile>"""
    with open(os.path.join(work, "column.vtu"), "w") as fh:
        fh.write(vtu)
    gml = f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<OpenGeoSysGLI xmlns:ogs="http://www.opengeosys.org">
<name>column_geometry</name>
<points><point id="0" x="0" y="0" z="0" name="BOTTOM"/><point id="1" x="{L}" y="0" z="0" name="TOP"/></points>
</OpenGeoSysGLI>"""
    with open(os.path.join(work, "column.gml"), "w") as fh:
        fh.write(gml)
    # 같은 vG 곡선을 표로
    h = -np.logspace(-4, 2, 120)[::-1]
    pc = -h * RHO_G
    Se = vg_Se(h, s["alpha"], s["n"])
    sat = (s["theta_r"] + (s["theta_s"] - s["theta_r"]) * Se) / s["theta_s"]
    kr = vg_kr(h, s["alpha"], s["n"])
    order = np.argsort(pc)
    pc_s, sat_s = pc[order], sat[order]
    so = np.argsort(sat)
    prj = f"""<?xml version='1.0' encoding='ISO-8859-1'?>
<OpenGeoSysProject>
 <mesh>column.vtu</mesh><geometry>column.gml</geometry>
 <processes><process><name>RF</name><type>RICHARDS_FLOW</type><integration_order>2</integration_order>
  <process_variables><process_variable>pressure</process_variable></process_variables>
  <secondary_variables><secondary_variable name="saturation"/></secondary_variables>
  <specific_body_force>-9.81</specific_body_force><mass_lumping>true</mass_lumping></process></processes>
 <media><medium id="0"><phases><phase><type>AqueousLiquid</type><properties>
   <property><name>viscosity</name><type>Constant</type><value>{mu}</value></property>
   <property><name>density</name><type>Constant</type><value>1000</value></property></properties></phase></phases>
  <properties>
   <property><name>storage</name><type>Constant</type><value>0</value></property>
   <property><name>porosity</name><type>Constant</type><value>{s['theta_s']}</value></property>
   <property><name>permeability</name><type>Constant</type><value>{k_int:.6e}</value></property>
   <property><name>relative_permeability</name><type>Curve</type><independent_variable>liquid_saturation</independent_variable><curve>kr</curve></property>
   <property><name>saturation</name><type>Curve</type><independent_variable>capillary_pressure</independent_variable><curve>pc</curve></property>
   <property><name>reference_temperature</name><type>Constant</type><value>293.15</value></property>
  </properties></medium></media>
 <time_loop><processes><process ref="RF"><nonlinear_solver>nl</nonlinear_solver>
   <convergence_criterion><type>DeltaX</type><norm_type>NORM2</norm_type><abstol>1e-8</abstol></convergence_criterion>
   <time_discretization><type>BackwardEuler</type></time_discretization>
   <time_stepping><type>FixedTimeStepping</type><t_initial>0</t_initial><t_end>3e7</t_end>
    <timesteps><pair><repeat>300</repeat><delta_t>1e5</delta_t></pair></timesteps></time_stepping></process></processes>
  <output><type>VTK</type><prefix>column_out</prefix><compress_output>false</compress_output>
   <timesteps><pair><repeat>1</repeat><each_steps>300</each_steps></pair></timesteps>
   <variables><variable>pressure</variable></variables></output></time_loop>
 <parameters>
  <parameter><name>p0</name><type>Constant</type><value>0</value></parameter>
  <parameter><name>p_bottom</name><type>Constant</type><value>0</value></parameter>
  <parameter><name>p_top</name><type>Constant</type><value>{h_top * RHO_G:.6f}</value></parameter></parameters>
 <curves>
  <curve><name>pc</name><coords>{' '.join(f'{v:.6e}' for v in pc_s)}</coords><values>{' '.join(f'{v:.8f}' for v in sat_s)}</values></curve>
  <curve><name>kr</name><coords>{' '.join(f'{v:.8f}' for v in sat[so])}</coords><values>{' '.join(f'{v:.6e}' for v in kr[so])}</values></curve>
 </curves>
 <process_variables><process_variable><name>pressure</name><components>1</components><order>1</order>
  <initial_condition>p0</initial_condition><boundary_conditions>
   <boundary_condition><geometrical_set>column_geometry</geometrical_set><geometry>BOTTOM</geometry><type>Dirichlet</type><parameter>p_bottom</parameter></boundary_condition>
   <boundary_condition><geometrical_set>column_geometry</geometrical_set><geometry>TOP</geometry><type>Dirichlet</type><parameter>p_top</parameter></boundary_condition>
  </boundary_conditions></process_variable></process_variables>
 <nonlinear_solvers><nonlinear_solver><name>nl</name><type>Newton</type><max_iter>50</max_iter><linear_solver>ls</linear_solver></nonlinear_solver></nonlinear_solvers>
 <linear_solvers><linear_solver><name>ls</name><eigen><solver_type>SparseLU</solver_type><scaling>true</scaling></eigen></linear_solver></linear_solvers>
</OpenGeoSysProject>"""
    prj_path = os.path.join(work, "column.prj")
    with open(prj_path, "w") as fh:
        fh.write(prj)

    def read_result(wdir):
        cand = sorted(f for f in os.listdir(wdir) if f.startswith("column_out") and f.endswith(".vtu"))
        if not cand:
            raise RuntimeError("OGS 출력 VTU를 찾지 못함")
        txt = open(os.path.join(wdir, cand[-1])).read()
        m = re.search(r'Name="pressure"[^>]*format="ascii"[^>]*>([^<]+)<', txt)
        if not m:
            raise RuntimeError("ASCII pressure 배열을 읽지 못함(압축/바이너리 출력)")
        p = np.array(m.group(1).split(), float)
        return x, p / RHO_G

    def fixed_top(g):
        fh = np.full(g.shape, np.nan)
        fh[0, 0, 0] = h_top
        return fh

    return {"prj": prj_path, "nz": nz, "L": L, "read_result": read_result, "fixed_top": fixed_top}
