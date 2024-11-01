from calendar import c
import os
import sys
import optparse
# 以上Python的标准库，用于文件处理、时间操作、命令行参数解析等。

import traci  # SUMO提供的控制接口，用于与仿真进行交互，获取交通仿真数据。
from traci import simulation, vehicle, chargingstation, lane, edge
from sumolib import checkBinary # sumolib ： SUMO工具库，用于辅助执行SUMO仿真任务。
import random
import numpy as np
import time
import copy
import csv
import pandas as pd
import tensorflow as tf

# 下面用于处理XML文件和优先队列。
from xml.dom import minidom
import heapq

# 这段代码主要与交通模拟、路径规划和延迟优化等功能有关。它利用SUMO（Simulation of Urban MObility）进行交通仿真，结合了机器学习（TensorFlow模型）以及常见的最短路径算法（如Dijkstra和Yen's K最短路径算法）。
def calculate_route_time(route): # 路径时间计算 （通过获取路线上的每个边缘（道路段）的行驶时间，计算整条路径的总行驶时间。）
    total_time = 0
    for i in range(len(route) - 1):
        edge_id = route[i]
        edge_length = traci.edge.getTraveltime(edge_id)
        total_time += edge_length
    return total_time

def dataWrite(method, repeat, v, d, charge, running, calcul, writer, AvgLinkDelay, AvgMaxDelay): # 该函数将仿真中的统计信息（例如运行时间、充电时间、计算时间等）汇总并写入CSV文件。
    print("평균 소요시간: {} simulation seconds".format(sum(running)/v))
    print("평균 충전시간: {} simulation seconds".format(sum(charge)/v))
    print("평균 계산시간: {} simulation seconds".format(sum(calcul)/v))
    temp_data=[]
    temp_data.append(method)
    temp_data.append(repeat)
    temp_data.append(v)
    temp_data.append(d)
    temp_data.append(sum(charge)/v)
    temp_data.append(sum(running)/v)
    temp_data.append(sum(calcul)/v)
    temp_data.append(sum(AvgLinkDelay)/len(AvgLinkDelay))
    temp_data.append(sum(AvgMaxDelay)/len(AvgMaxDelay))
    print('Algorithm, laps, Vehicle number, Destination number, Charging Time, Running Time, Calculation Time, Average Link Delay, Average Max Delay')
    print(temp_data)
    final_data=[]
    final_data.append(temp_data)
    writer.writerows(final_data)

#  用于解析命令行选项，决定是否使用SUMO的GUI界面。
def get_options(args=None):
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    print("options:", options, "args:", args) # options: {'nogui': False} args: []
    return options

# 找出字典中最小值的键。
def get_key_with_min_value(dictionary):
    min_value = min(dictionary.values())
    for key, value in dictionary.items():
        if value == min_value:
            return key
        
        
def Link_info_update(edge_list, AvgLinkDelay, AvgMaxDelay):
    #edge_list 자체가 결과 리스트의 순서와 동일
    perfo_data=[]
    for ed in edge_list:
        if edge.getTraveltime(ed):
            perfo_data.append(edge.getTraveltime(ed))
        else:
            perfo_data.append(0)
    
    
    AvgLinkDelay.append(sum(perfo_data)/len(perfo_data))
    if max(perfo_data)!=66400:
        AvgMaxDelay.append(max(perfo_data))


# edge_info_update_saint 和 edge_info_update 的区别是saint中还考虑了CCM平均值的影响。
def edge_info_update_saint(edge_list, ccm_list, AvgLinkDelay, AvgMaxDelay):
    #edge_list 자체가 결과 리스트의 순서와 동일
    final_data=[]
    for ed in edge_list:
        if edge.getTraveltime(ed):
            final_data.append(edge.getTraveltime(ed))
        else:
            final_data.append(0)
    
    output=dict()
    AvgLinkDelay.append(sum(final_data)/len(final_data))
    if max(final_data)!=66400:
        AvgMaxDelay.append(max(final_data))
    
    for i in range(len(edge_list)):
        CCMavrg=sum(ccm_list[edge_list[i]])/len(ccm_list[edge_list[i]])
        if CCMavrg==0:
            output[edge_list[i]]=final_data[i] # 반영할 CCM이 없으면 곱해주지 않음
        else:
            output[edge_list[i]]=final_data[i]*(1+CCMavrg) # 반영할 CCM이 있으면 y_pred 값에 곱해주기
    
    return output

# 边缘信息更新
# 这些函数用于更新交通仿真中道路段（edge）的状态信息。调用SUMO接口获取交通数据（如等待时间、平均速度、车辆数等），然后使用一个预训练的机器学习模型（TensorFlow）进行延迟预测。
def edge_info_update(edge_list, y_pred, ccm_list):
    #edge_list 자체가 결과 리스트의 순서와 동일
    final_data=[]
    for ed in edge_list:
        temp=[]
        if edge.getPendingVehicles(ed):
            temp.append(len(edge.getPendingVehicles(ed)))
        else:
            temp.append(0)
            
        if edge.getWaitingTime(ed):
            temp.append(edge.getWaitingTime(ed))
        else:
            temp.append(0)
            
        if edge.getLastStepMeanSpeed(ed):
            temp.append(edge.getLastStepMeanSpeed(ed))
        else:
            temp.append(0)
            
        if edge.getLastStepVehicleNumber(ed):
            temp.append(edge.getLastStepVehicleNumber(ed))
        else:
            temp.append(0)
            
        final_data.append(temp)
        # data = [['Pending Vehicles', 'Waiting Time','Last Step Mean Speed', 'Last Step Vehicle Number']]
    
    loaded_model = tf.keras.models.load_model("Multi_LR_Model.h5")
    y_pred = loaded_model.predict(final_data)
    output=dict()
    
    for i in range(len(edge_list)):
        CCMavrg=sum(ccm_list[edge_list[i]])/len(ccm_list[edge_list[i]])
        if CCMavrg==0:
            output[edge_list[i]]=y_pred[i] # 반영할 CCM이 없으면 곱해주지 않음
        else:
            output[edge_list[i]]=y_pred[i]*(1+CCMavrg) # 반영할 CCM이 있으면 y_pred 값에 곱해주기
    
    return output

# dijkstra算法  （实现了经典的Dijkstra算法用于计算图中从起点到其他节点的最短距离。）
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0 
    predecessors = {node: None for node in graph}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue) 
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


# 根据前驱节点信息找到特定目标节点的最短路径。
def find_shortest_path(predecessors, target):
    path = []
    current_node = target
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]
    return path

# 更新CCM信息 (updateCCM, deleteCCM)
# CCM表示"交通拥堵管理"的相关信息，主要用于记录车辆在道路上的拥堵情况。updateCCM 根据当前车辆的位置更新该信息，而deleteCCM 则移除车辆的CCM记录。
def updateCCM(shortest_path, cur_loca, ccm_list, vehicle_ccm, curVehicle):
    for k in shortest_path:
        vehicle_ccm[curVehicle][k]=0 # {edge: 0, edge:0, ... , edge:0} default setting
    
    totalTime=simulation.findRoute(cur_loca, shortest_path[-1]).travelTime
    del shortest_path[-1]
    
    for i in shortest_path:
        tempratio=1-(simulation.findRoute(cur_loca, i).travelTime/totalTime)
        ccm_list[i].append(tempratio)
        vehicle_ccm[curVehicle][i]=tempratio

# 移除车辆的CCM记录。
def deleteCCM(curVehicle, ccm_list, vehicle_CCM):
    for ed in vehicle_CCM[curVehicle]:
        if vehicle_CCM[curVehicle][ed]!=0:
            ccm_list[ed].remove(vehicle_CCM[curVehicle][ed])
            
    vehicle_CCM[curVehicle].clear()

# 随机选择一个充电站，确保它不在车辆的当前道路上。
def search_qcm(charging_staion_list, vehicle_id):
    a = 1
    while a == 1:
        random_qcm = random.choice(charging_staion_list)
        if lane.getEdgeID(chargingstation.getLaneID(random_qcm["charging_station_id"])) != vehicle.getRoadID(vehicle_id):
            a = 0
    return random_qcm

# 深度优先搜索算法，用于解决旅行商问题，找到访问所有节点的最短路径。
def dfs(start, next, value, visited, N, travel, min_value, path):
    if min_value[0]<value:
        return
    if len(visited) == N:
        # 모든 목적지 방문을 완료한 경우
        if travel[next][start][0] != 0:
            ori=min_value[0] #원래 최소 비용
            min_value[0] = min(min_value[0], value + travel[next][start][0])
            if ori!=min_value[0]:
                path[0]=visited.copy()
        return
        
    for i in range(N):
        if travel[next][i] != 0 and i != start and i not in visited:
            visited.append(i)
            dfs(start, i, value + travel[next][i][0], visited, N, travel, min_value, path)
            visited.pop()

 # Yen's K最短路径算法
 # 用于在路径规划时找到多个最短路径。在找到第一条最短路径后，通过修改图结构找到接下来的K条次优路径。该算法会排除掉某些边以计算次优路径。
def yen_k_shortest_paths(dijk_data, start_node, k, destination_info_g, curVehicle):
    k_shortest_paths = []
    for _ in range(k):
        if not k_shortest_paths:
            # Compute the shortest path for the first iteration
            distances, predecessors = dijkstra(dijk_data, start_node)
            options = {}
            for destination_node in destination_info_g[curVehicle]:
                options[destination_node] = distances[destination_node]

            next_target = get_key_with_min_value(options)
            shortest_path = find_shortest_path(predecessors, next_target)
            k_shortest_paths.append(shortest_path)
        else:
            # Compute the next shortest path by excluding edges from previous paths
            last_path = k_shortest_paths[-1]
            for i in range(len(last_path) - 1):
                spur_node = last_path[i]
                root_path = last_path[:i + 1]

                removed_edges = []
                for path in k_shortest_paths:
                    if len(path) > i and root_path == path[:i + 1]:
                        edge_to_remove = (path[i], path[i + 1])
                        dijk_data[path[i]].pop(path[i + 1])
                        removed_edges.append(edge_to_remove)

                spur_distances, spur_predecessors = dijkstra(dijk_data, spur_node)
                if next_target in spur_distances:
                    spur_path = find_shortest_path(spur_predecessors, next_target)
                    total_path = root_path + spur_path[1:]

                    k_shortest_paths.append(total_path)

                # Restore removed edges
                for edge_to_restore in removed_edges:
                    dijk_data[edge_to_restore[0]][edge_to_restore[1]] = distances[edge_to_restore[0]][edge_to_restore[1]]

        if not k_shortest_paths:
            break

    return k_shortest_paths

# 计算给定路径的总成本
def calculate_path_cost(path, dijk_data): # path: 包含节点的列表，表示车辆行驶路径。 dijk_data: 使用Dijkstra算法计算的图数据结构，存储节点之间的边的成本。
    path_cost = 0
    for i in range(len(path) - 1): # 遍历路径中的所有相邻节点，计算每条边的成本。
        edge_cost = dijk_data[path[i]][path[i + 1]]  # Assuming dijk_data stores edge costs
        path_cost += edge_cost
    return path_cost


# 检查是否设置了SUMO_HOME环境变量，并将SUMO的工具路径添加到Python路径中。
if 'SUMO_HOME' in os.environ: # os.environ 是存储环境变量的字典。
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools')) # 如果存在 SUMO_HOME 变量，代码将其下的工具路径添加到 sys.path，使得后续可以调用SUMO相关的库和工具。

np.random.seed(0) # 设置随机数生成器的种子，以确保后续生成的随机数是可预测和可复现的。设置种子 0 确保每次运行程序时，生成的随机数序列相同.


def run(vehiclenum, destination_number, SAINT_writer, repeat): # (车辆数量， 每辆车的目的地数量， 用于记录输出结果的对象， 控制实验重复运行的次数)
    detour_factor=0.1 # 绕行系数，可能用于计算车辆行驶的偏差。 (这个参数可能表示车辆在行驶过程中绕行的比例，用于模拟道路封闭或交通拥堵导致的车辆行驶偏差。绕行系数越高，车辆的实际路径偏离预期路径的程度越大。绕行10%：如果车辆要从起点到达终点，理论上有一条最短路径。然而，由于各种原因，车辆可能无法沿着最短路径行驶，而是需要绕行。绕行系数0.1意味着，车辆可能偏离最短路径的 10%，也就是说，它行驶的距离会比理想最短路径多出10%。)
    step = 0 # 初始化仿真步数。
    battery_full_amount = 1500 # 车辆电池的最大容量。
    charging_station_list = [] # 初始化充电站列表。里面会加入很多字典， 每个字典就是一个充电站的信息(充电站ID， 站内等候车辆)
    edge_list = [] # 初始化边列表，保存交通网络中的有效边。
    ccm_list = dict() # 初始化车辆的CCM（车辆累计成本？）字典。 (CCM可能指的是某种累积成本（Cumulative Cost Metric），表示车辆在行驶过程中所产生的费用、时间、能量消耗等累计值。该字典用于记录每辆车的累积成本。)
    vehicle_CCM=list() # 初始化每辆车的CCM列表。
    edge_connection={} # 初始化边连接字典，用于存储边的连接信息（from -> to）。
    AvgLinkDelay=[] # 初始化平均链路延迟列表。
    AvgMaxDelay=[] # 初始化最大链路延迟列表。
    # xfile=minidom.parse('demo.net.xml')
    xfile=minidom.parse('Gangnam4.net.xml') # 解析网络配置文件 Gangnam4.net.xml，获取交通网络的结构信息。 (minidom.parse() 函数会读取 XML 文件的内容，将其解析成一个 DOM (Document Object Model) 对象)
    connections=xfile.getElementsByTagName('connection') # 获取所有交通网络中的连接元素（<connection>）节点，表示边的起点和终点。构建边连接字典 edge_connection。

    # # 遍历每个 <connection> 元素并打印其信息
    # for conn in connections:
    #     # 获取 'from' 和 'to' 属性值
    #     from_edge = conn.getAttribute('from')
    #     to_edge = conn.getAttribute('to')
    #
    #     # 打印连接信息
    #     print(f'Connection from {from_edge} to {to_edge}')
    #
    # input("---------------------------------------------------暂停------------------------------------------------------")



    # 处理交通网络连接信息
    for c in connections: # 遍历所有的连接 c
        if c.attributes['from'].value in edge_connection: # 检查 edge_connection 字典中是否有 from节点的键名，如果有添加 to 节点到该节点的邻居列表。
            edge_connection[c.attributes['from'].value].append(c.attributes['to'].value)
        else: #
            edge_connection[c.attributes['from'].value]=[c.attributes['to'].value] # 如果没有， 则创建一个列表并将 to 节点添加进去， 这个列表叫邻居列表。
    # 得到了所有的connection节点的连接信息
    
    for k,v in edge_connection.items(): # 遍历出所有键值对
        edge_connection[k]=list(set(v)) # 중복 제거  # 对每个 from 节点的邻居列表， to节点进行去重，防止重复连接到同一节点。
    
    
    # matrix 구조 : [vehicle number][출발지][travel time, 도착지]
    # 목적지+출발지

    # 初始化充电站和边信息
    for charging_station in chargingstation.getIDList(): # chargingstation.getIDList() 返回仿真中的所有充电站ID
        # 为每个充电站创建一个字典 temp_charging_station_dict，包含 charging_station_id（充电站ID）和 waiting_vehicle（等待充电的车辆列表）。

        temp_charging_station_dict = {}
        temp_charging_station_dict = {
            "charging_station_id": charging_station,
            "waiting_vehicle": np.array([])
        }
        charging_station_list.append(temp_charging_station_dict) # 将充电站信息添加到 充电站列表charging_station_list中去。

    # 遍历所有的边， 然后确保 边是和所有充电站所在的边都不一样， 然后把过滤后的有效的边的id 添加进edge_list中，还有 做成 边ID:[0]的键值对放进ccm_list字典中
    for temp_edge in edge.getIDList(): # 遍历所有边 temp_edge，返回边缘ID， 如果边不是信号边（以 : 开头），且不位于充电站所在的边上，将该边添加到 edge_list 并初始化其CCM值为 0。过滤掉充电站所在的边，初始化非充电站边的 CCM 值为 0，并将这些边加入到 edge_list。
        for charging_station_dict in charging_station_list:#遍历所有的充电站
            chgstnPos_=lane.getEdgeID(chargingstation.getLaneID(charging_station_dict["charging_station_id"])) # 通过字典获取当前充电站的 ID。 通过充电站 ID 获取充电站所在的车道 ID。 通过车道 ID 获取充电站对应的边缘 ID（chgstnPos_）。
            if temp_edge[0] != ":" and temp_edge != chgstnPos_: # 确保 temp_edge 不是以 ":" 开头的无效 ID  and  确保当前边缘 ID 不等于充电站位置的边缘 ID。
                edge_list.append(temp_edge) # 将当前有效的边缘 ID (temp_edge) 添加到 edge_list 列表中
                ccm_list[temp_edge]=[0] # 在字典 ccm_list 中为当前边缘 ID (temp_edge) 创建一个键，并初始化其值为一个包含零的列表（[0]）。
                break
            else:
                break


    # 初始化车辆目的地信息
    destination_info=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)] # 배송지만 포함  生成一个大小为 vehiclenum × destination_number 的二维列表， 每个元素为0
    destination_info_g=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)] # 배송지만 포함
    tot_destination_info=[[0 for _ in range(destination_number+1)] for _ in range(vehiclenum)] # 배송지+출발지  포함

    time_info=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)] # 保存每辆车的时间信息。  大小为 vehiclenum × destination_number 的二维列表， 每个元素为0
    for i in range(0, vehiclenum):
        destination_info[i]=random.sample(edge_list,destination_number) # 随机为每辆车分配 destination_number 个目的地，从有效边 edge_list 中随机抽取 destination_number 个元素， 返回一个列表
    destination_info_g = copy.deepcopy(destination_info) # 并且使用 deepcopy 复制目的地信息，以便后续修改不会影响原始数据。

    # 初始化车辆状态和信息
    car_index = 0
    finish_info=[0 for _ in range(vehiclenum)] # 保存每辆车已完成的目的地数量。

    # 分别保存每辆车的总时间和临时时间。
    time_list=[0.0 for _ in range(vehiclenum)]
    time_temp=[0.0 for _ in range(vehiclenum)]

    # 保存每辆车充电相关信息。
    charge_temp=[0.0 for _ in range(vehiclenum)]
    charge_info=[0.0 for _ in range(vehiclenum)]

    # 保存每辆车计算相关信息。
    calcul_temp=[0.0 for _ in range(vehiclenum)]
    calcul_info=[0.0 for _ in range(vehiclenum)]
    
    step=0
    time_info=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)] #  destination_number * vehiclenum 二维列表， 用于存储每辆车到达每个目的地的时间信息
    destination_info_g=copy.deepcopy(destination_info) # 深拷贝：会复制 destination_info 中所有的元素，且对其中的嵌套对象进行递归复制。这样即使修改 destination_info_g，也不会影响到 destination_info。创建一个与 destination_info 内容完全相同但相互独立的副本。
    # 三维列表， 外层列表长度为 vehiclenum，表示车辆的数量。对于每辆车，内层是一个长度为 destination_number 的列表，其中每个元素是 [0, i]。
    # [0, i] 中，0 可能是表示初始距离或时间，而 i 表示目的地编号（从 0 到 destination_number-1）。
    fromStart=[[[0,i] for i in range(destination_number)] for _ in range(vehiclenum)]  # 初始化一个三维列表，用于存储每辆车从起始点到每个目的地的相关信息，例如到达时间或距离。

    # 向系统中添加了 vehiclenum 辆电动车，并为每辆车分配了唯一的车辆 ID 和默认路线。
    for i in range(0, vehiclenum):
        #在 vehicle 对象上调用 add 方法，将一辆新车添加到系统中。为每辆车分配一个唯一的 ID。给车辆分配的路线 ID 为 "route_0"，即每辆车都使用相同的路线。typeID='ElectricCar'：车辆类型为 "ElectricCar"，代表这是电动车。 车辆类型为 "ElectricCar"，代表这是电动车。departPos=0：车辆的出发位置设置为 0，通常表示起点。
        vehicle.add(vehID="vehicle_" + str(car_index), routeID="route_0", typeID='ElectricCar', departPos=0) #  为每辆车添加一个带有唯一 ID、指定路线、类型和出发位置的车辆信息。
        car_index += 1
     # 为每辆车在 vehicle_CCM 列表中创建了一个空字典，可能用于后续存储车辆的自定义信息。
    for _ in range(vehiclenum):
        vehicle_CCM.append(dict()) # 在 vehicle_CCM 列表中追加一个空字典 dict()。用于存储每辆车的相关数据或信息。


    # 主循环
    while simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        

        # 车辆状态管理
        for vehicle_id in vehicle.getIDList(): # 获取当前正在仿真中的所有车辆ID。
            # 车辆初始化（为每辆车分配 vehicle_id，并为其设置出发点、目的地、电池容量等属性。为每辆车初始化一个 CCM 字典，记录其行驶状态。）
            curVehicle=int(vehicle_id.split('_')[1]) # split('_') 方法将 vehicle_id 按照下划线 '_' 进行分割，返回一个列表。例如，如果 vehicle_id 是 'vehicle_123'，那么 split('_') 的结果是 ['vehicle', '123']。
            if vehicle.getParameter(vehicle_id, "state") == "": # 如果车辆状态为空，说明该车辆刚刚加入仿真或尚未分配任何任务，初始化车辆状态、实际电池容量和目的地。
                vehicle.setParameter(vehicle_id, "actualBatteryCapacity", 1500)
                vehicle.setParameter(vehicle_id, "state", "running") # 设置状态为运行
                
                for Dstnt in range(destination_number):
                    # fromStart 是一个三维数组。存储了当前车辆到每个目的地的行驶时间和目的地信息。
                    # destination_info_g[curVehicle][Dstnt] 提供当前车辆到目的地的信息（例如目的地的ID或位置）。 simulation.findRoute() 函数用于找到从当前车辆所在的道路到目的地的最佳路径，返回一个包含路径信息的对象。
                    # 通过simulation.findRoute().travelTime函数计算从当前车辆到每个目的地的行驶时间，并将该时间存储在fromStart[curVehicle][Dstnt][0]中。
                    fromStart[curVehicle][Dstnt][0]=simulation.findRoute(vehicle.getRoadID(vehicle_id), destination_info_g[curVehicle][Dstnt]).travelTime
                    fromStart[curVehicle][Dstnt][1]=destination_info_g[curVehicle][Dstnt] # 将目的地的信息存储在fromStart[curVehicle][Dstnt][1]中。通常是目的地的ID、坐标等。

                # 1. 对目的地进行排序，选择行驶时间最短的目的地。
                # 2. 记录当前时间，并为车辆设置新的目的地。
                # 3. 改变车辆的目标为新的目的地，并清空车辆的下一个充电站参数。
                fromStart[curVehicle].sort(key=lambda x:x[0])  # 对fromStart[curVehicle] 根据行驶时间排序多个目的地,因为x[0] 是从车辆到目的地的行驶时间
                time_temp[curVehicle]=simulation.getTime() # 通过simulation.getTime()函数记录当前的模拟时间，并将其存储在time_temp[curVehicle]中。
                vehicle.setParameter(vehicle_id, "destination", fromStart[curVehicle][0][1])  # 为当前车辆设置目的地。 fromStart[curVehicle][0][1] 是行驶时间最短的目的地信息（因为前面已经按行驶时间排序）。通过这行代码，将最短行驶时间对应的目的地设定为当前车辆的目标目的地。
                vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination")) #  获取当前车辆的目的地参数，该参数之前已通过上一步设置为行驶时间最短的目的地。
                vehicle.setParameter(vehicle_id, "next_charging_station", "")  # 将车辆的下一个充电站参数清空。将值设为""（空字符串），意味着当前没有设定下一个充电站

            else:
                # 车辆正在运行
                if vehicle.getParameter(vehicle_id, "state") == "running":
                    
                    battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))   # 获取当前电池量
                    battery_amount -= 1    # 电池量减少1
                    vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)   # 更新电池量
                    
                    # 判断是否需要充电
                    nowBatt=float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                    needBatt=simulation.findRoute(vehicle.getRoadID(vehicle_id), vehicle.getParameter(vehicle_id, "destination")).travelTime+25  # 充电所需电量

                    if vehicle.getParameter(vehicle_id, "next_charging_station") == "" and nowBatt < needBatt: # 如果电量不足
                        time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle]  # 计算当前行驶时间
                        time_list[curVehicle]+=time_temp[curVehicle]

                        # 计算所需的参数
                        calcul_temp[curVehicle]=(-1)*time.time()
                        # ADDED START
                        pair_output=edge_info_update_saint(edge_list, ccm_list, AvgLinkDelay, AvgMaxDelay) # 通过训练模型更新信息来计算预期输出值
                        dijk_data=dict()
                        
                        for key, values in edge_connection.items():
                            temp_values={}
                            for t in values:
                                if t in pair_output:
                                    temp_values[t]=pair_output[t]
                            dijk_data[key]=temp_values
                            
                        start_node=vehicle.getRoadID(vehicle_id)
                        distances, predecessors = dijkstra(dijk_data, start_node) # 在电量不足时，车辆会通过 Dijkstra 算法计算到最近充电站的最短路径，并沿该路径行驶。
                        
                        options=dict()
                        options_pos=dict()
                        
                        for ch in chargingstation.getIDList(): # 查找下一个充电站
                            chstnPos=lane.getEdgeID(chargingstation.getLaneID(ch))
                            options[chstnPos]=distances[chstnPos]
                            options_pos[chstnPos]=ch
                        
                        next_target=get_key_with_min_value(options)
                        next_target_chstation_id=options_pos[next_target]
                        
                        vehicle.changeTarget(vehicle_id, next_target)
                        if next_target!=start_node:
                            shortest_path = find_shortest_path(predecessors, next_target)
                            vehicle.setRoute(vehicle_id, shortest_path)
                            vehicle.setChargingStationStop(vehicle_id, next_target_chstation_id)
                            
                        vehicle.setParameter(vehicle_id, "next_charging_station", next_target_chstation_id) # vehicle의 next charging station 값을 바꿔줌
                        
                        calcul_temp[curVehicle]+=time.time()
                        calcul_info[curVehicle]+=calcul_temp[curVehicle] # 계산 끝
                        time_temp[curVehicle]=simulation.getTime() # restart
                        

                    # 如果车辆到达充电站，则将状态设置为“等待”，并更新车辆的累积成本（CCM）。
                    if vehicle.getParameter(vehicle_id, "next_charging_station") and vehicle.getLaneID(vehicle_id)==chargingstation.getLaneID(vehicle.getParameter(vehicle_id, "next_charging_station")):
                        time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle] # 충전소 도착했으니, running time 멈춤
                        time_list[curVehicle]+=time_temp[curVehicle]
                        vehicle.setParameter(vehicle_id, "state", "waiting")
                        deleteCCM(curVehicle, ccm_list, vehicle_CCM)
                        # DONE 충전소에 도착했으면, 해당 CCM 빼주기 

                    # CCM管理：车辆在到达充电站或目的地后，CCM会被重新计算和更新。
                    # 路径更新：车辆会根据实时计算的最短路径更新其行驶路线。
                        
                    # 当车辆到达其目的地后，如果还有未到达的目的地，车辆会通过更新交通网络的边信息来计算下一个最近的目的地，并继续行驶。如果车辆已经完成所有目的地的配送任务，它将停止运行。
                    if vehicle.getRoadID(vehicle_id) == vehicle.getParameter(vehicle_id, "destination"):
                        if vehicle.getParameter(vehicle_id, "next_charging_station") == "":
                            # 귀환 위치가 아닌 배송지에 도착한 경우에만 info update & 이번 도착지를 dest list에서 삭제 
                            # DONE 한 목적지에 도착했으면, 해당 CCM 빼주기 
                            deleteCCM(curVehicle, ccm_list, vehicle_CCM)
                            time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle]
                            time_list[curVehicle]+=time_temp[curVehicle]
                            
                            if finish_info[curVehicle] < destination_number:
                                time_info[curVehicle][int(finish_info[curVehicle])] = step
                                finish_info[curVehicle] += 1
                                
                            nowGoal=vehicle.getParameter(vehicle_id,"destination")
                            destination_info_g[curVehicle].remove(nowGoal)
                            
                            if len(destination_info_g[curVehicle])>0: # 아직 가야할 목적지가 남아있는 경우
                                # 남은 목적지 리스트에서 가장 가까운 목적지 검색
                                # non-model, only saint
                                pair_output=edge_info_update_saint(edge_list, ccm_list, AvgLinkDelay, AvgMaxDelay) # update the information to compute the expected output value through trained model
                                dijk_data=dict()
                                
                                for key, values in edge_connection.items():
                                    temp_values={}
                                    for t in values:
                                        # print("this is key: {}, this is values: {}, and this is T: {}".format(key, values, t))
                                        if t in pair_output:
                                            temp_values[t]=pair_output[t]
                                    dijk_data[key]=temp_values
                                
                                start_node = nowGoal
                                calcul_temp[curVehicle]=(-1)*time.time()
                                
                                k=3
                                removed_edges = []
                                k_shortest_paths = []
                                
                                while(len(k_shortest_paths)<k):
                                    if not k_shortest_paths:
                                        # Compute the shortest path for the first iteration
                                        distances, predecessors = dijkstra(dijk_data, start_node)
                                        options = dict()
                                        for destination_node in destination_info_g[curVehicle]:
                                            options[destination_node] = distances[destination_node]

                                        next_target = get_key_with_min_value(options) #첫번째 다익스트라의 결과로 첫번째 목적지는 정해짐
                                        shortest_path = find_shortest_path(predecessors, next_target) # 첫번째 목적지로 가는 shortest path 계산
                                        k_shortest_paths.append(shortest_path)
                                        
                                    else:
                                        for path in k_shortest_paths:
                                            for i in range(1, len(path)-2, 3): # len에서 2 빼는 이유, 인덱스 and i+1 인덱스 연산해야 하기 때문
                                                if i<=len(path)-2 and (path[i+1] in dijk_data[path[i]]):
                                                    edge_to_remove = (path[i], path[i + 1], dijk_data[path[i]][path[i + 1]])
                                                    dijk_data[path[i]].pop(path[i + 1])
                                                    removed_edges.append(edge_to_remove)
                                                    # print("removed_edges {}".format(removed_edges))
                                                    
                                        # FIND 사용된 edge없이 dijkstra cost 계산
                                        spur_distances, spur_predecessors = dijkstra(dijk_data, start_node)
                                        if next_target in spur_distances: # 새로운 dijkstra cost 목록에 next target이 있는가? 
                                            spur_path = find_shortest_path(spur_predecessors, next_target) # 있다면, 그 목적지로 가는 2nd shortest path 찾어
                                            k_shortest_paths.append(spur_path)

                                        
                                    if len(k_shortest_paths)>=k:
                                        
                                        # k개의 경로 찾기 완료 
                                        # 삭제되었던 edge 다시 restore 해주기 
                                        for edge_to_restore in removed_edges:
                                            # print("edge to restore: {}".format(edge_to_restore))
                                            dijk_data[edge_to_restore[0]][edge_to_restore[1]] = edge_to_restore[2]
                                        # print("I'M SO DONE, path num is {}".format(len(k_shortest_paths)))
                                        
                                        
                                        # k개의 경로 중 어떤 것으로 선택할 지 결정 
                                        baseline=calculate_route_time(k_shortest_paths[0])
                                        saveID=0
                                        for p in range(1, k, 1):
                                            if len(k_shortest_paths[p])!=1 and calculate_route_time(k_shortest_paths[p])<baseline*(1+detour_factor):
                                                saveID=p
                                                # print("shortest PATH is changed")
                                        
                                        if saveID!=0:
                                            shortest_path=copy.deepcopy(k_shortest_paths[saveID])
                                        #break

                                    #if not shortest_path:
                                        #break

                                
                                updateCCM(shortest_path, nowGoal, ccm_list, vehicle_CCM, curVehicle)
                                traci.vehicle.setRoute(vehicle_id, shortest_path)
                                
                                # 남은 목적지 중 가장 가까운 곳으로 destination 설정
                                vehicle.setParameter(vehicle_id, "destination", next_target)
                                vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                                calcul_temp[curVehicle]+=time.time()
                                calcul_info[curVehicle]+=calcul_temp[curVehicle]
                                
                                time_temp[curVehicle]=simulation.getTime() # restart, 주행시간 측정 다시 시작

                            else: # 배송 완료
                                vehicle.remove(vehicle_id)
                                # print("vehicle_{} 배송 완료, 차량 {}대 중 배송이 남은 차량 {}대".format(curVehicle, vehiclenum, vehicle.getIDCount()))
                                if vehicle.getIDCount()==0:
                                    print("FINISH")
                                    if destination_number==1:
                                        Link_info_update(edge_list, AvgLinkDelay, AvgMaxDelay)
                                    dataWrite("SAINT", repeat, vehiclenum, destination_number, charge_info, time_list, calcul_info, SAINT_writer, AvgLinkDelay, AvgMaxDelay)

                
                # vehicle 상태가 waiting이었는데 자기 차례가 이제 됐다면, waiting to charging
                elif vehicle.getParameter(vehicle_id, "state") == "waiting" and vehicle.getStopState(vehicle_id) == 65:
                    charge_temp[curVehicle]=(-1)*simulation.getTime() # 충전 시간 측정 시작
                    vehicle.setParameter(vehicle_id, "state", "charging")
                    
                    
                # vehicle 상태가 Charging이면
                elif vehicle.getParameter(vehicle_id, "state") == "charging":
                    battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                    battery_amount += 4
                    vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)

                    # 배터리 양이 최대치로 충전 됐을 때, 충전중인 자동차들 다시 목적지로 출발
                    if battery_amount >= battery_full_amount:
                        charge_temp[curVehicle]+=simulation.getTime()
                        charge_info[curVehicle]+=charge_temp[curVehicle] # 충전 시간 측정 끝
                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                        vehicle.setParameter(vehicle_id, "state", "running")
                        vehicle.setParameter(vehicle_id, "next_charging_station", "")
                        time_temp[curVehicle]=simulation.getTime() # restart
                        vehicle.resume(vehicle_id)
                        
        step += 1
      
    traci.close()
    sys.stdout.flush()
    

if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        print("没有GUI")
        sumoBinary = checkBinary('sumo') # 通过checkBinary('sumo')来选择不带GUI的SUMO二进制文件
    else:
        print("有GUI")#  运行!!!
        sumoBinary = checkBinary('sumo-gui') # 通过checkBinary('sumo-gui')来选择带GUI的SUMO二进制文件。

    # 打开一个CSV文件用于记录结果数据。如果文件不存在则会创建它。newline=''是为了防止在Windows系统中写入空行。
    SAINT_file=open('only_saint_more_perfo.csv', 'w', newline='')

    # 定义了一行数据，用于记录  [算法名称、圈数、车辆数量、目的地数量、充电时间、运行时间、计算时间、平均链路延迟、平均最大延迟]
    entry = [['Algorithm', 'laps', 'Vehicle number', 'Destination number', 'Charging Time', 'Running Time', 'Calculation Time', 'Average Link Delay', 'Average Max Delay']]

    # 创建一个csv.writer对象，用于将数据写入CSV文件。
    SAINT_writer = csv.writer(SAINT_file)

    # 将entry（包含标题行的数据）写入CSV文件。
    SAINT_writer.writerows(entry)
            
    # Destination and Vehicle Number diff
    
    for d in range(0, 10, 1): # d表示目的地数量，每次增加1。
        for repeat in range(1, 11, 1): # 表示仿真运行的重复次数，每次重复10次。
            vNum=50 # 每次仿真的车辆数量设定为50。
            if d==0:
                dNum=1 # 目的地数量dNum设定为1。
            else:
                dNum=d*5

        #  dNum: 1, 5, 10, 15, 20, 25, 30, 35, 40, 45
        #repeat: 1, 2,  3,  4,  5,  6,  7,  8,  9, 10
            try:
                traci.start([sumoBinary, "-c", "Gangnam4.sumocfg"]) # 启动SUMO仿真，使用选定的二进制文件sumoBinary和配置文件Gangnam4.sumocfg。traci是SUMO的Python接口，用于控制SUMO仿真过程。
                # traci.start([sumoBinary, "-c", "demo_30.sumocfg"])
                start = time.time() # 记录仿真开始时间。
                run(vNum, dNum, SAINT_writer, repeat) # 调用run函数，传递 (车辆数量vNum, 目的地数量dNum, CSV写入器SAINT_writer记录结果, 重复次数repeat), 运行仿真并记录结果。
                
            except Exception: # 捕获并忽略在仿真过程中发生的任何异常，以确保程序即使发生错误也不会崩溃。
                pass
