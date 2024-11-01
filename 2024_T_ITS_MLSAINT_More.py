from calendar import c
import os
import sys
import optparse
from turtle import circle
import traci
from traci import simulation, vehicle, chargingstation, lane, edge, route, constants
from sumolib import checkBinary
import random
import numpy as np
import time
import copy
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from xml.dom import minidom
import heapq
import csv



def calculate_route_time(route):
    total_time = 0
    for i in range(len(route) - 1):
        edge_id = route[i]
        edge_length = traci.edge.getTraveltime(edge_id)
        total_time += edge_length
    return total_time

def dataWrite(method, repeat, v, d, charge, running, calcul, writer, AvgLinkDelay, AvgMaxDelay):
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

def get_options(args=None):
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def get_key_with_min_value(dictionary):
    min_value = min(dictionary.values())
    for key, value in dictionary.items():
        if value == min_value:
            return key


def Link_info_update(edge_list, AvgLinkDelay, AvgMaxDelay):
    #edge_list 자체가 결과 리스트의 순서와 동일 ##edge_list itself is the same as the order of the result list
    perfo_data=[]
    for ed in edge_list:
        if edge.getTraveltime(ed):
            perfo_data.append(edge.getTraveltime(ed))
        else:
            perfo_data.append(0)
    
    
    AvgLinkDelay.append(sum(perfo_data)/len(perfo_data))
    if max(perfo_data)!=66400:
        AvgMaxDelay.append(max(perfo_data))

def edge_info_update(edge_list, y_pred, ccm_list, AvgLinkDelay, AvgMaxDelay):
    #edge_list 자체가 결과 리스트의 순서와 동일  #edge_list itself is the same as the order of the result list
    print("edge_info_update的 edge_list的长度是：", len(edge_list)) # 1414
    final_data=[]#是一个二维的数组， 里面每一个小数据都会包含一条edge的四个特征值
    perfo_data=[]
    for ed in edge_list:
        temp=[]
        print("第一步")
        if edge.getPendingVehicles(ed): # 获取当前道路上等待的车辆列表（ed 表示当前道路的 ID）
            temp.append(len(edge.getPendingVehicles(ed))) # 车辆数量
        else:
            temp.append(0) # 当前道路上没有等待的车辆。

        print("第二步")
        if edge.getWaitingTime(ed): # 获取当前道路上车辆的等待时间（单位通常是秒）， 如果该值存在，表示有车辆在该道路上等待，并且有具体的等待时间，将此值添加到 temp。
            temp.append(edge.getWaitingTime(ed))
        else:
            temp.append(0) # 没有等待时间

        print("第四步")
        if edge.getLastStepVehicleNumber(ed):  # 获取当前道路上行驶的车辆数量
            temp.append(edge.getLastStepVehicleNumber(ed))
        else:
            temp.append(0)

        print("第三步")
        if edge.getLastStepMeanSpeed(ed): # 获取当前道路上车辆的平均速度（单位通常是 m/s 或 km/h）。
            temp.append(edge.getLastStepMeanSpeed(ed))
        else:
            temp.append(0) # 如果没有该信息（例如没有车辆在这条路上行驶），则向 temp 添加 0。

        print("第五步， 获取标签")
        if edge.getTraveltime(ed): # 获取当前道路的旅行时间（单位通常是秒）， 如果旅行时间存在，表示车辆通过该道路的时间
            perfo_data.append(edge.getTraveltime(ed))
        else:
            perfo_data.append(0)

        print("第六步")
        final_data.append(temp)
        # data = [['Pending Vehicles', 'Waiting Time','Last Step Mean Speed', 'Last Step Vehicle Number']]
    final_data = np.array(final_data)
    print("final_data的类型是：", type(final_data), "形状是:", final_data.shape) # final_data的类型是： <class 'list'> 形状是: (1414, 4)

    # loaded_model = tf.keras.models.load_model("Multi_LR_Model_adam.h5")
    loaded_model = tf.keras.models.load_model("./model/previous/Multi_LR_Model_adam.h5",custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})
    print("success！！！")

    y_pred = loaded_model.predict(final_data) # 对所有道路的交通信息进行预测，预测结果存储在 y_pred 中。 接调用 model.predict() 方法不会改变模型内的权重参数。无论是在训练模式还是评估模式下，调用 predict 都只是进行前向传播计算，而不会对模型的权重进行更新或修改。

    print("得到的预测值的形状是：", y_pred.shape) # (1414, 1)
    # input("-----------------------------------------------------------暂停-----------------------------------------------------------")
    #创建返回值字典
    output=dict()
    AvgLinkDelay.append(sum(perfo_data)/len(perfo_data)) # 计算当前所有道路的 平均旅行时间（AvgLinkDelay）

    if max(perfo_data)!=66400: # 如果最大延迟值不为 66400（可能是仿真中的一个极限值或无效值），则将最大延迟添加到 AvgMaxDelay 中。
        AvgMaxDelay.append(max(perfo_data))
    
    for i in range(len(edge_list)): # 遍历每条道路，计算该道路的 平均拥堵系数（CCMavrg）
        CCMavrg = sum(ccm_list[edge_list[i]])/len(ccm_list[edge_list[i]]) # 计算该道路的 平均拥堵系数（CCMavrg）

        if CCMavrg==0: # CCMavrg 为 0，表示没有拥堵系数影响，则直接使用预测值 y_pred[i]。
            output[edge_list[i]]=y_pred[i] # 每条edge对应延迟预测值
        else: # 如果 CCMavrg 大于 0，则将预测值 y_pred[i] 乘以 (1 + CCMavrg)，即考虑拥堵系数对预测结果的影响。
            output[edge_list[i]] = y_pred[i] * (1 + CCMavrg)

    print("edge_info_update函数执行完成！")
    return output # 返回值， 每条edge对应的延迟预测值

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


def find_shortest_path(predecessors, target):
    path = []
    current_node = target
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]
    return path

def updateCCM(shortest_path, cur_loca, ccm_list, vehicle_ccm, curVehicle):
    for k in shortest_path:
        vehicle_ccm[curVehicle][k]=0 # {edge: 0, edge:0, ... , edge:0} default setting
    
    totalTime=simulation.findRoute(cur_loca, shortest_path[-1]).travelTime
    del shortest_path[-1]
    
    for i in shortest_path:
        tempratio=1-(simulation.findRoute(cur_loca, i).travelTime/totalTime)
        ccm_list[i].append(tempratio)
        vehicle_ccm[curVehicle][i]=tempratio
    
def deleteCCM(curVehicle, ccm_list, vehicle_CCM):
    for ed in vehicle_CCM[curVehicle]:
        if vehicle_CCM[curVehicle][ed]!=0:
            ccm_list[ed].remove(vehicle_CCM[curVehicle][ed])
            
    vehicle_CCM[curVehicle].clear()
        

def search_qcm(charging_staion_list, vehicle_id):
    a = 1
    while a == 1:
        random_qcm = random.choice(charging_staion_list)
        if lane.getEdgeID(chargingstation.getLaneID(random_qcm["charging_station_id"])) != vehicle.getRoadID(vehicle_id):
            a = 0
    return random_qcm


def dfs(start, next, value, visited, N, travel, min_value, path):
    if min_value[0]<value:
        return
    if len(visited) == N:
        # 모든 목적지 방문을 완료한 경우 # When all destinations have been visited
        if travel[next][start][0] != 0:
            ori=min_value[0] #원래 최소 비용  #Original minimum cost
            min_value[0] = min(min_value[0], value + travel[next][start][0])
            if ori!=min_value[0]:
                path[0]=visited.copy()
        return
        
    for i in range(N):
        if travel[next][i] != 0 and i != start and i not in visited:
            visited.append(i)
            dfs(start, i, value + travel[next][i][0], visited, N, travel, min_value, path)
            visited.pop()
            


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



def calculate_path_cost(path, dijk_data):
    path_cost = 0
    for i in range(len(path) - 1):
        edge_cost = dijk_data[path[i]][path[i + 1]]  # Assuming dijk_data stores edge costs
        path_cost += edge_cost
    return path_cost

           
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
np.random.seed(0)

def run(vehiclenum, destination_number, ML_file, Greedy_file, BnB_file, repeat):
    stop_duration = 60 # 停车时间设为 60 秒
    detour_factor=0.1 # 绕行系数，用于可能的路径调整。
    step = 0 # 表示仿真中的当前步数
    battery_full_amount = 1500 # 电池的最大充电量
    charging_station_list = [] # 存储充电站信息的列表。里面储存多个字典， 每个字典都是关于充电站的信息
    edge_list = [] # 存储网络中边的信息（即可能的行驶路径）。
    ccm_list = dict() # 字典，用来记录每个edge的状态。
    vehicle_CCM=list() # 列表，保存每辆车的充电控制模块（CCM）状态。
    edge_connection={} # 存储网络中每个边的连接关系（从一个边到另一个边的连接）
    AvgLinkDelay=[] # 每辆车在每条路径上遇到的平均链接延迟和最大延迟。
    AvgMaxDelay=[]

    shortest_path_b=[[0] for _ in range(vehiclenum)] # 初始化一个二维列表，用来存储每辆车的最短路径信息，默认每辆车的路径为 [0]。
    final_path=[[0 for _ in range(destination_number+1)] for _ in range(vehiclenum)] # 初始化一个二维列表，用于保存每辆车访问各个目的地的最终路径，大小为 vehiclenum x (destination_number+1)，其中 +1 包括出发点。
    matrix=[[[[0,i] for i in range(destination_number+1)] for _ in range(destination_number+1)] for _ in range(vehiclenum)] # 四维列表，保存车辆之间的目的地信息和各路径的出发、到达时间等。

    #创建三个csv写入器
    ML_writer = csv.writer(ML_file) # 机器学习算法 性能数据
    Greedy_writer = csv.writer(Greedy_file) # 贪婪算法 性能数据
    BnB_writer = csv.writer(BnB_file) # 分支定界算法 性能数据


    # 解析读取仿真网络的拓扑结构。
    xfile=minidom.parse('Gangnam4.net.xml')
    connections=xfile.getElementsByTagName('connection')#从文件中获取所有的连接元素，表示网络中各个边之间的连接关系。
    print("The number of connections is :", len(connections)) # 8815


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

    # 将每条边的起点和终点加入 edge_connection 字典中，并移除重复的连接。
    for c in connections:
        if c.attributes['from'].value in edge_connection:
            edge_connection[c.attributes['from'].value].append(c.attributes['to'].value)
        else:
            edge_connection[c.attributes['from'].value]=[c.attributes['to'].value] #这里把该键对应的值 设置成为了列表， 所以上面可以通过append进行添加
    for k,v in edge_connection.items():
        edge_connection[k]=list(set(v)) # 중복 제거  #remove duplicates

    # 打印每个节点的连接关系, 因为都是有向边， 所以有起点是-117785727#1， 终点是-117785727#0的边的话， 并不代表一定有起点是 -117785727#0， 终点是-117785727#1的边
    print("字典edge_connection的长度是 : ", len(edge_connection)) # 6139
    for k, v in edge_connection.items():
        print(f"边{k}的连接节点有{len(v)}个， 分别是：{v}")
    # input("---------------------------------------------------暂停------------------------------------------------------")
    

    # matrix 구조 : [vehicle number][출발지][travel time, 도착지]
    # 遍历网络中的所有充电站，初始化每个充电站的 ID 和等待的车辆列表。
    for charging_station in chargingstation.getIDList():
        temp_charging_station_dict = {} #每次创建一个新词典
        temp_charging_station_dict = {
            "charging_station_id": charging_station,
            "waiting_vehicle": np.array([])
        }
        charging_station_list.append(temp_charging_station_dict)
    # print(f"充电站的个数是: {len(charging_station_list)}") # 52，  在Gangnam4.add.xml文件中进行定义的
    # print(charging_station_list)
    # input("---------------------------------------------------暂停------------------------------------------------------")

    print("edge的总数是：", len(edge.getIDList()), "返回值类型是:", type(edge.getIDList())) # 6142 ,<class 'tuple'>, 里面都是edge的ID值

    # 遍历所有的边，然后把过滤后的有效的边的id 添加进edge_list中，还有 做成 边ID:[0]的键值对放进ccm_list字典中
    wuXiao = 0
    for temp_edge in edge.getIDList(): #每一个temp_edge都是一个id值
        for charging_station_dict in charging_station_list:
            chgstnPos_=lane.getEdgeID(chargingstation.getLaneID(charging_station_dict["charging_station_id"]))  # 通过字典获取当前充电站的 ID。 通过充电站 ID 获取充电站所在的车道Lane ID。 通过车道 ID 获取充电站对应的边缘edga ID（chgstnPos_）。
            # print("充电站的ID是：", charging_station_dict["charging_station_id"], ", 充电站所在edga的ID是：", chgstnPos_) # edga ID是： 417627646#0

            if temp_edge[0] != ":" and temp_edge != chgstnPos_: # 确保 temp_edge 不是以 ":" 开头的无效 ID  and  确保当前边缘 ID 没有任何充电站
                edge_list.append(temp_edge)
                ccm_list[temp_edge]=[0]
                break
            else:
                wuXiao += 1
                break
    print("edge_list的总长度是:", len(edge_list)) #1414 (名字对， 而且没有任何充电站)
    print("无效边：", wuXiao) # 4728， 和上面的141加在一i去正好等于 edge的总数6142


    # 初始化每辆车的目的地信息二维列表 (大小为 vehiclenum × destination_number 的二维列表， 每个元素为0)
    destination_info=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)] # 배송지만 포함，  生成一个大小为 vehiclenum × destination_number 的二维列表， 每个元素为0
    # destination_info_g=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)] # 배송지만 포함

    # 初始化每辆车的目的地信息二维列表， 包括出发点。 (大小为 vehiclenum × destination_number + 1 的二维列表)
    tot_destination_info=[[0 for _ in range(destination_number+1)] for _ in range(vehiclenum)] # 배송지+출발지

    # 保存每辆车的时间信息。 （ 大小为 vehiclenum × destination_number 的二维列表）
    time_info=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)]

    # 随机为每辆车分配 destination_number 个目的地(第一次run时dNum是1)，random.sample是从有效边 edge_list 中随机抽取 destination_number 个元素， 返回一个列表
    for i in range(0, vehiclenum):
        destination_info[i]=random.sample(edge_list,destination_number)
    destination_info_g = copy.deepcopy(destination_info) # 并且使用 deepcopy 复制目的地信息，以便后续修改不会影响原始数据。
    print("destination_info:", destination_info_g) # 50个小列表， 每个小列表都有 dNum个 edge ID字符串

    # input("-----------------------------------------------------------------------------------------------------------------暂停")

    # 初始化不同的列表（如 finish_info、time_list、charge_info 等），这些用于跟踪车辆的完成状态、时间、充电等信息。
    for iterNum in range(1): # 只执行1次
        car_index = 0
        finish_info=[0 for _ in range(vehiclenum)] # 实时储存每个车辆完成的目标数
        time_list=[0.0 for _ in range(vehiclenum)] # 总行驶时间列表
        time_temp=[0.0 for _ in range(vehiclenum)] # 部分路段的行驶时间
        charge_temp=[0.0 for _ in range(vehiclenum)]
        charge_info=[0.0 for _ in range(vehiclenum)]
        calcul_temp=[0.0 for _ in range(vehiclenum)]
        calcul_info=[0.0 for _ in range(vehiclenum)]
        
        step=0
        time_info=[[0 for _ in range(destination_number)] for _ in range(vehiclenum)]
        destination_info_g=copy.deepcopy(destination_info) # 复制 destination_info，生成每次仿真车辆的目的地列表。
        fromStart=[[[0,i] for i in range(destination_number)] for _ in range(vehiclenum)] # 三维数组， vehiclenum * destination_number * [0,i], 比如有2辆车， 每辆车有3个目的地时：[  [[0, 0], [0, 1], [0, 2]],    [[0, 0], [0, 1], [0, 2]]  ]
        print("fromStart: ", fromStart)
            
        if iterNum==0: # ML Linear Regression Model
            print("Machine Learning")
            # 配置每一辆车
            for i in range(0, vehiclenum):
                # 每辆车都有唯一的 vehID，且类型为 'ElectricCar'，并且它们会按照给定的路线 route_0 进行仿真。车辆的出发位置 (departPos)，设置为 0。
                vehicle.add(vehID="vehicle_" + str(car_index), routeID="route_0", typeID='ElectricCar', departPos=0) # 使用 vehicle.add() 添加电动汽车到仿真中，车辆类型为 ElectricCar。
                car_index += 1

            # 为每辆车创建一个充电控制模块（CCM）的字典，以记录每辆车的状态。
            for _ in range(vehiclenum):
                vehicle_CCM.append(dict())

            #进入主循环
            while simulation.getMinExpectedNumber() > 0:
                traci.simulationStep() # 这是一个用于推进 SUMO 交通仿真的单步操作的函数。每调用一次，仿真时间将前进一个时间步（通常是一个固定的时间间隔，例如 1 秒），并且所有车辆和交通信号等都会根据当前状态更新。这是一个用于推进 SUMO 交通仿真的单步操作的函数。每调用一次，仿真时间将前进一个时间步（通常是一个固定的时间间隔，例如 1 秒），并且所有车辆和交通信号等都会根据当前状态更新。


                #遍历每个车辆， 给每个车辆按照 耗费时间 重新排列目的地列表
                print("这一步的车辆列表是：", vehicle.getIDList()) #每一步的车辆列表里的车辆个数都不一样， 是一个累加的过程
                input("----------------------------------------------------")
                for vehicle_id in vehicle.getIDList(): # 通过 vehicle.getIDList() 获取所有车辆的列表，并循环处理每辆车的状态。
                    curVehicle=int(vehicle_id.split('_')[1]) # 只获取当前车辆名字中的数字ID， 因为上面创建每辆车的名字时， 格式是 vehicle_1

                    # 如果车辆还未初始化
                    if vehicle.getParameter(vehicle_id, "state") == "": # 如果该参数为空，说明车辆还未初始化。
                        print("车辆还未初始化， 开始初始化车辆状态信息！") # 50个
                        vehicle.setParameter(vehicle_id, "actualBatteryCapacity", 1500) # 设定电池容量为1500
                        vehicle.setParameter(vehicle_id, "state", "running") # 并将车辆状态设置为 'running'


                        # 当前车辆  开始计算 配送包裹到自己的每个目的地的时间
                        for Dstnt in range(destination_number):
                            # fromStart 是一个三维数组。存储了当前车辆到 该目的地的行驶时间和目的地ID。 simulation.findRoute(当前车辆所在道路ID， 目的地道路ID), 但是当前车辆所在道路ID都是-417746801
                            # findRoute 返回的路径对象中包含 travelTime 属性，表示从起始道路到目的地的预计行驶时间。
                            fromStart[curVehicle][Dstnt][0]=simulation.findRoute(vehicle.getRoadID(vehicle_id), destination_info_g[curVehicle][Dstnt]).travelTime
                            fromStart[curVehicle][Dstnt][1]=destination_info_g[curVehicle][Dstnt] # 得到的是目的地节点ID， 219835825#4， 车辆编号

                        #到目前为止 得到了 当前车辆到每个目的地的[行驶时间, 目的地ID]

                        #下面给 当前车辆车设置最近目的地
                        fromStart[curVehicle].sort(key=lambda x:x[0]) #  然后按照行驶时间对其进行排序， 选出当前车辆的最近目的地
                        time_temp[curVehicle]=simulation.getTime() # 记录当前车的此刻时间， time_temp是一维列表
                        vehicle.setParameter(vehicle_id, "destination", fromStart[curVehicle][0][1])  # 为当前的车辆（由 vehicle_id 标识）设置一个目的地，该目的地是从 fromStart 数据结构中提取的车辆的目标位置。 fromStart[curVehicle][0]是当前车辆的最近目的地，[1]是该目标节点ID
                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination")) # 获取某辆车当前的目的地信息，然后将该目的地设置为该车的目标
                        vehicle.setParameter(vehicle_id, "next_charging_station", "") # 将下一个充电站设置为空， 表示当前车辆暂时没有指定的下一个充电站。可能是因为车辆在当前行程中不需要马上充电，或者下一个充电站还没有确定。

                    #如果已经初始化了，并处于 'running' 状态的车辆
                    else:
                        if vehicle.getParameter(vehicle_id, "state") == "running":
                            # print("车辆在运行！")
                            battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))  # 得到当前电量
                            battery_amount -= 1  # 逐步减少其电池容量（每一步减少1单位）
                            vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)  # 赋予更新后的电量
                            
                            # 배터리의 양이 Threshold일때
                            nowBatt=float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                            needBatt=simulation.findRoute(vehicle.getRoadID(vehicle_id), vehicle.getParameter(vehicle_id, "destination")).travelTime+25 # 找到车辆从当前道路到目标道路的路径(当前车辆当前所在的道路ID, 获取车辆当前的目的地).travelTime：路径的预期行驶时间， 在预期行驶时间的基础上增加25个单位时间，可能是为了预留安全的电池余量。

                            # 如果车辆还没有配置充电站    and    电池容量不足以到达目标位置，则执行充电逻辑。
                            if vehicle.getParameter(vehicle_id, "next_charging_station") == "" and nowBatt < needBatt: #부족 #lack
                                time_temp[curVehicle] = simulation.getTime() - time_temp[curVehicle] # 计算从车辆出发到现在的时间差，停止对行驶时间的记录。（再一次从仿真系统中获取当前时间 - 前存储的时间点）
                                time_list[curVehicle] += time_temp[curVehicle] # 将 当前车辆的行驶时间差 加入到总时间列表中。 time_list是一维列表
                                print("NEED CHARGE!!!")
                                
                                calcul_temp[curVehicle]=(-1)*time.time() # 开始记录计算路径所花的时间，记录当前时间戳，乘以 -1， 下面会进行+=计算，用于计算时间差。   calcul_temp是一维列表

                                y_pred=[]
                                pair_output = edge_info_update(edge_list, y_pred, ccm_list, AvgLinkDelay, AvgMaxDelay) #返回值是一个字典， 预测的每条edge对应的延迟时间

                                # 将模型预测的边缘数据转化为适合 Dijkstra 算法的数据结构，用于计算最短路径。
                                dijk_data=dict() # 用于 Dijkstra 算法的数据字典，起始边id : {终点边id:延迟值，………… 终点边id:延迟值}

                                #遍历所有边(图中的边都是有向边)的连接关系字典，如果一条边的 终点在pair_output中， 就添加到temp_values字典中， 格式是 边id:延迟值
                                for key, values in edge_connection.items(): # key：边的起点，   values: 边的终点们
                                    temp_values={}
                                    for t in values:
                                        if t in pair_output:
                                            temp_values[t]=pair_output[t] # 从模型预测结果中获取每条边的权重，更新 dijk_data。
                                    dijk_data[key]=temp_values
                                    
                                start_node=vehicle.getRoadID(vehicle_id)
                                distances, predecessors = dijkstra(dijk_data, start_node)# 使用 Dijkstra 算法计算从车辆当前所在位置 (start_node) 到各个节点的最短路径。 得到1. 从当前节点到各个节点的最短距离。 2. 每个节点在最短路径中的前驱节点，用于追踪最短路径。


                                # 在所有充电站中找到距离车辆当前位置最近的充电站
                                options=dict()
                                options_pos=dict()
                                for ch in chargingstation.getIDList(): # 获取所有充电站的ID列表。
                                    chstnPos=lane.getEdgeID(chargingstation.getLaneID(ch)) # 根据充电站的位置获取所在道路的ID。
                                    options[chstnPos]=distances[chstnPos] # 将充电站的道路ID与其距离对应
                                    options_pos[chstnPos]=ch
                                
                                next_target=get_key_with_min_value(options) # 找到距离最短的充电站。
                                next_target_chstation_id=options_pos[next_target] # 得到此充电站的id


                                # 更新车辆的目标和路径
                                vehicle.changeTarget(vehicle_id, next_target) # 将车辆的目标位置更新为最近的充电站。
                                if next_target!=start_node:
                                    shortest_path = find_shortest_path(predecessors, next_target) # 根据 Dijkstra 算法的前驱节点追踪最短路径。
                                    vehicle.setRoute(vehicle_id, shortest_path) # 设置车辆的行驶路径。
                                    vehicle.setChargingStationStop(vehicle_id, next_target_chstation_id) # 为车辆设置目标充电站。
                                vehicle.setParameter(vehicle_id, "next_charging_station", next_target_chstation_id) # 更新车辆的“下一个充电站”参数

                                # 记录路径规划的计算时间并重新开始计时，继续跟踪车辆的行驶时间。
                                calcul_temp[curVehicle]+=time.time() # 得到时间差，也就是行驶时间， 储存进calcul_temp[curVehicle]中，
                                calcul_info[curVehicle]+=calcul_temp[curVehicle] # 将计算时间累加到总的计算时间列表中。 时间计算结束
                                time_temp[curVehicle]=simulation.getTime() # 重新开始计时，用于后续记录行驶时间。
                                

                            # 车辆到达充电站时的处理。 （判断车辆是否有下一个充电站的目标 and 检查车辆是否已经到达目标充电站所在的车道）
                            if vehicle.getParameter(vehicle_id, "next_charging_station") and vehicle.getLaneID(vehicle_id)==chargingstation.getLaneID(vehicle.getParameter(vehicle_id, "next_charging_station")):
                                print("车辆到达 充电站！")
                                time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle] # 到达充电站，计算从上次记录到现在的时间，并更新临时时间变量。
                                time_list[curVehicle]+=time_temp[curVehicle]  # 将这段时间加到车辆的总行驶时间
                                vehicle.setParameter(vehicle_id, "state", "waiting")# 设置车辆状态为“等待”
                                deleteCCM(curVehicle, ccm_list, vehicle_CCM)  # 删除当前车辆的CCM
                                # DONE 충전소에 도착했으면, 해당 CCM 빼주기  # DONE When you arrive at the charging station, remove the CCM
                                
                                
                            # 车辆到达目的地时的处理 (当前车辆所在道路ID   ==  车辆目的道路ID)
                            if vehicle.getRoadID(vehicle_id) == vehicle.getParameter(vehicle_id, "destination"): # 两者相等时，说明车辆已经到达目标
                                print("车辆到达 目的道路！") # 49个， 46个， 48个
                                if vehicle.getParameter(vehicle_id, "next_charging_station") == "": # 判断是否有充电站目标，如果没有，则表示车辆正在进行任务
                                    time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle] # 计算从上次记录到现在的时间，并更新临时时间变量。
                                    time_list[curVehicle]+=time_temp[curVehicle]  #将到达目的地的运行时间累加到时间列表。 更新总行驶时间
                                    deleteCCM(curVehicle, ccm_list, vehicle_CCM) # 删除当前车辆的CCM

                                    # 如果车辆完成的目标数小于总目标数
                                    if finish_info[curVehicle] < destination_number: # finish_info是一维列表
                                        time_info[curVehicle][int(finish_info[curVehicle])] = step # 将当前的时间步（step）记录到当前车辆的当前目的地 time_info 列表中,表示在该时间步，当前车辆到达了此目的地，  这一步之前time_info[][]的对应位置是0
                                        finish_info[curVehicle] += 1 # 更新已完成的目标数量
                                        
                                    nowGoal=vehicle.getParameter(vehicle_id,"destination") # 获取当前目标位置
                                    destination_info_g[curVehicle].remove(nowGoal)# 从目标列表(之前随机生成的)中移除当前目标。

                                    # 如果还有目的地需要到达， 就选择新的目的地
                                    if len(destination_info_g[curVehicle])>0:
                                        # 在剩下的目的地中找到最近的目的地
                                        y_pred=[]
                                        pair_output=edge_info_update(edge_list, y_pred, ccm_list, AvgLinkDelay, AvgMaxDelay) # 调用 edge_info_update 函数，通过模型预测来更新边的信息（如链路延迟）。

                                        # 构建Dijkstra算法所需的数据结构
                                        dijk_data=dict()
                                        for key, values in edge_connection.items(): #遍历边与节点的连接关系，将边权重信息存储到 dijk_data 中。
                                            temp_values={}
                                            for t in values:
                                                if t in pair_output:
                                                    temp_values[t]=pair_output[t] #从模型预测结果中获取每条边的权重，更新 dijk_data。
                                            dijk_data[key]=temp_values

                                        # 使用 Dijkstra 算法从当前节点（目的地）开始，计算到下一个目标的最短路径。
                                        start_node = nowGoal
                                        calcul_temp[curVehicle]=(-1)*time.time()

                                        # 寻找k条最短路径（通过多次运行Dijkstra算法，寻找 k 条不同的最短路径，并尝试移除一些边来计算备用路径。最终通过比较路径耗时选择最合适的路径。）
                                        k=3
                                        removed_edges = []
                                        k_shortest_paths = []
                                        
                                        while(len(k_shortest_paths)<k):
                                            if not k_shortest_paths: # 如果是第一次迭代，则计算初始最短路径。
                                                distances, predecessors = dijkstra(dijk_data, start_node) # 调用Dijkstra算法，计算从起始节点到其他节点的最短距离。
                                                # 记录到每个目的地的距离。
                                                options = dict()
                                                for destination_node in destination_info_g[curVehicle]:
                                                    options[destination_node] = distances[destination_node]

                                                # 选择距离最近的目标节点。
                                                next_target = get_key_with_min_value(options)
                                                # 找到到达目标的最短路径，并将其添加到k_shortest_paths列表中。
                                                shortest_path = find_shortest_path(predecessors, next_target)
                                                k_shortest_paths.append(shortest_path)
                                                
                                            else: #如果不是第一次迭代
                                                for path in k_shortest_paths: # 遍历每条路径
                                                    for i in range(1, len(path)-2, 3): # len에서 2 빼는 이유, 인덱스 and i+1 인덱스 연산해야 하기 때문 # The reason for subtracting 2 from len is because we need to calculate the index and i+1 index.

                                                        if i<=len(path)-2 and (path[i+1] in dijk_data[path[i]]): # 确保当前遍历的节点 i 不是路径的最后一个节点，避免越界(因为要访问 path[i+1])，并且检查路径中 path[i] 和 path[i+1] 之间的边是否在图 dijk_data 中。
                                                            # dijk_data[path[i]]是一个字典，存储了从 path[i] 节点出发可以到达的所有邻居节点，以及到这些节点的边权重。如果path[i+1]在此字典中， 说明在 dijk_data 图中，path[i] 到 path[i+1] 是一条有效边，可以被移除。
                                                            edge_to_remove = (path[i], path[i + 1], dijk_data[path[i]][path[i + 1]]) # 将这条边记录下来， 做成一个三元组
                                                            dijk_data[path[i]].pop(path[i + 1]) # 从 dijk_data 图中移除该边。
                                                            removed_edges.append(edge_to_remove)# 将刚刚移除的边添加到 removed_edges 列表中，便于之后恢复。
                                                            
                                                # 使用剩余的图再次运行 Dijkstra：
                                                spur_distances, spur_predecessors = dijkstra(dijk_data, start_node)
                                                if next_target in spur_distances: # 새로운 dijkstra cost 목록에 next target이 있는가? # Is there a next target in the new dijkstra cost list? 
                                                    spur_path = find_shortest_path(spur_predecessors, next_target) # 있다면, 그 목적지로 가는 2nd shortest path 찾어
                                                    k_shortest_paths.append(spur_path)


                                            # 检查是否已经找到 k 条最短路径
                                            if len(k_shortest_paths)>=k: # 一旦找到了足够的路径（即 k 条路径）
                                                for edge_to_restore in removed_edges: # 遍历 removed_edges 列表，将之前从 dijk_data 图中移除的边重新添加回去
                                                    dijk_data[edge_to_restore[0]][edge_to_restore[1]] = edge_to_restore[2] #将边的信息重新加入到图中， 每个edge_to_restore都是 一个三元组列表，包含移除的边（起点、终点和边权重）
                                                
                                                # 在找到的 k 条路径中选择一条合适的路径
                                                baseline=calculate_route_time(k_shortest_paths[0]) # 计算第一条路径的时间，作为基线比较。
                                                saveID=0
                                                for p in range(1, k, 1): # 检查哪一条路径比基线路径更短
                                                    # 如果 本路径长度符合要求，且 比最短路径的1.1倍短
                                                    if len(k_shortest_paths[p])!=1 and calculate_route_time(k_shortest_paths[p])<baseline*(1+detour_factor): # 绕行因子(0.1)，允许一定范围的偏差
                                                        saveID=p #更新最短路径索引值
                                                        # print("shortest PATH is changed")

                                                # 如果 saveID 不为 0，表示找到了一条比基线更好的路径，将其作为最优路径保存
                                                if saveID!=0:
                                                    shortest_path=copy.deepcopy(k_shortest_paths[saveID])

                                        updateCCM(shortest_path, nowGoal, ccm_list, vehicle_CCM, curVehicle) # 更新车辆的 CCM， (最短路径， 当前目标节点， CCM 列表， 当前车辆的 CCM， 当前车辆的 ID)
                                        traci.vehicle.setRoute(vehicle_id, shortest_path) # 使用 TraCI API 将车辆的路线设置为最优路径。
                                        
                                        # 在剩下的目的地中 设置最近的目的地
                                        vehicle.setParameter(vehicle_id, "destination", next_target) # 设置车辆的目的地为下一个目标。
                                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination")) # 让车辆前往新的目标地点。

                                        # 记录计算和仿真时间
                                        calcul_temp[curVehicle]+=time.time()
                                        calcul_info[curVehicle]+=calcul_temp[curVehicle] # 계산시간 측정 끝 # End of calculation time measurement
                                        time_temp[curVehicle]=simulation.getTime() # 记录当前仿真时间，用于后续时间管理。

                                    else: # 배송 완료
                                        vehicle.remove(vehicle_id) # 车辆完成任务，移除该车辆
                                        if vehicle.getIDCount()==0: # 如果所有车辆都已完成任务，打印“FINISH”。
                                            print("FINISH!!")
                                            if destination_number==1:
                                                print("本次的destination_number值为1")
                                                Link_info_update(edge_list, AvgLinkDelay, AvgMaxDelay) # 更新链路信息（可能是记录链路延迟或最大延迟等数据）
                                                print("destination_number值为1， 完成计算") # ！！！！！！！！！！！！这里出现了问题
                                            dataWrite("ML_SAINT", repeat, vehiclenum, destination_number, charge_info, time_list, calcul_info, ML_writer, AvgLinkDelay, AvgMaxDelay) # 将数据写入文件，包括车辆信息、充电时间、计算时间等。

                        
                        # vehicle 상태가 waiting이었는데 자기 차례가 이제 됐다면, waiting to charging
                        elif vehicle.getParameter(vehicle_id, "state") == "waiting" and vehicle.getStopState(vehicle_id) == 65:
                            print("车辆开始等待， 进行充电！ 状态码为65")
                            charge_temp[curVehicle]=(-1)*simulation.getTime() # 开始记录充电时刻。 下面通过 += 运算记录充电时间
                            vehicle.setParameter(vehicle_id, "state", "charging") # 设置为‘充电’状态
                            
                            
                        # 车辆状态为“充电”时。
                        elif vehicle.getParameter(vehicle_id, "state") == "charging":
                            print("车辆正在充电！")
                            battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                            battery_amount += 6 # 每次迭代增加 6 单位的电池电量
                            vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)

                            # 当车辆的电池充满后，更新充电时间，并将车辆的状态从“充电”切换为“行驶”，清空充电站信息并重新开始行驶。
                            if battery_amount >= battery_full_amount: # 检查电池是否充满
                                print("车辆电池已充满电")
                                charge_temp[curVehicle] += simulation.getTime() # 得到充电时间
                                charge_info[curVehicle] += charge_temp[curVehicle] # 保存车辆的总充电时间
                                vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination")) # 让充满电的车辆重新设置目的地，准备继续行驶。   vehicle.getParameter(vehicle_id, "destination")：获取车辆的目的地参数，这里是设置的下一个目标。
                                vehicle.setParameter(vehicle_id, "state", "running") # 恢复车辆的运行状态
                                vehicle.setParameter(vehicle_id, "next_charging_station", "") # 将车辆的“下一个充电站”参数设置为空，意味着车辆在本段行驶中不再计划去充电站。
                                vehicle.resume(vehicle_id) # 恢复车辆的运行状态，重新开始移动。通常在车辆因为充电、停车或其他原因暂停后，需要调用该方法让车辆重新上路。
                                time_temp[curVehicle]=simulation.getTime() # 记录当前车辆重新开始行驶的时刻
                                
                step += 1
                print("step ", step)
            print("ML中 while循环 完成！")

        elif iterNum==1: # Greedy
            print("Greedy Algorithm")
            for i in range(0, vehiclenum):
                vehicle.add(vehID="vehicle_" + str(car_index), routeID="route_0", typeID='ElectricCar', departPos=0) # 每个车辆被分配一个唯一的 ID (vehicle_" + str(car_index))，并且初始化在 route_0 路线
                car_index += 1
            while simulation.getMinExpectedNumber() > 0:
                traci.simulationStep() # 通过 traci.simulationStep() 逐步推进模拟。
                for vehicle_id in vehicle.getIDList():
                    
                    # 초기화 코드  # initialization code
                    # vehicle state가 running이 아니면,
                    curVehicle=int(vehicle_id.split('_')[1]) # 获取每辆车的 ID 并通过检查 state 参数，确认是否已经初始化。如果车辆状态为空，则将其电池容量设定为 1500 并设置状态为 "running"。
                    if vehicle.getParameter(vehicle_id, "state") == "":
                        vehicle.setParameter(vehicle_id, "actualBatteryCapacity", 1500)
                        vehicle.setParameter(vehicle_id, "state", "running")
                        
                        for Dstnt in range(destination_number): # 为每个目的地计算从当前车辆位置到目标的预计行驶时间，并将这些信息存储在 fromStart 数组中。
                            fromStart[curVehicle][Dstnt][0]=simulation.findRoute(vehicle.getRoadID(vehicle_id), destination_info_g[curVehicle][Dstnt]).travelTime
                            fromStart[curVehicle][Dstnt][1]=destination_info_g[curVehicle][Dstnt]

                        
                        fromStart[curVehicle].sort(key=lambda x:x[0]) # 对所有目的地进行排序，以选择离当前车辆最近的目的地作为目标。
                        time_temp[curVehicle]=simulation.getTime()
                        vehicle.setParameter(vehicle_id, "destination", fromStart[curVehicle][0][1])  # vehicle_1 <- 1 extract, -> vehicle 1은 1 destination info를 선택

                        # 目标设置完成后，车辆将其目标位置更新为选定的目的地，并且改变行驶目标。
                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                        vehicle.setParameter(vehicle_id, "next_charging_station", "")

                    else:
                        # vehicle 상태가 Running이면
                        if vehicle.getParameter(vehicle_id, "state") == "running":
                            battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))  # 현재 값을 가져와서  # Get the current value
                            battery_amount -= 1  # 车辆在行驶时，每一步都消耗1个电池电量
                            vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)  # 줄인 값을 다시 할당  # Reassign the reduced value
                            
                            # 배터리의 양이 Threshold일때  # When the battery amount is Threshold
                            nowBatt=float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                            needBatt=simulation.findRoute(vehicle.getRoadID(vehicle_id), vehicle.getParameter(vehicle_id, "destination")).travelTime+25 # ???????????????????????

                            if vehicle.getParameter(vehicle_id, "next_charging_station") == "" and nowBatt < needBatt: #当车辆电量不足且未分配下一个充电站时，开始寻找合适的充电站
                                time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle]
                                time_list[curVehicle]+=time_temp[curVehicle]
                                
                                calcul_temp[curVehicle]=(-1)*time.time()
                                min_expect_time = 100000000
                                min_expect_qcm = search_qcm(charging_station_list, vehicle_id)

                                for charging_station_dict in charging_station_list:
                                    
                                    vehPos=vehicle.getRoadID(vehicle_id)
                                    chstnPos=lane.getEdgeID(chargingstation.getLaneID(charging_station_dict["charging_station_id"]))
                                    needBattToChstn=simulation.findRoute(vehicle.getRoadID(vehicle_id),lane.getEdgeID(chargingstation.getLaneID(charging_station_dict["charging_station_id"]))).travelTime

                                    if (vehPos != chstnPos) and (nowBatt > needBattToChstn):
                                        min_expect_time = simulation.findRoute(vehicle.getRoadID(vehicle_id), lane.getEdgeID(chargingstation.getLaneID(min_expect_qcm["charging_station_id"]))).travelTime

                                        if min_expect_time >= needBattToChstn:
                                            min_expect_qcm = charging_station_dict

                                vehicle.changeTarget(vehicle_id, lane.getEdgeID(chargingstation.getLaneID(min_expect_qcm["charging_station_id"])))
                                vehicle.setChargingStationStop(vehicle_id, min_expect_qcm["charging_station_id"])
                                vehicle.setParameter(vehicle_id, "next_charging_station", min_expect_qcm["charging_station_id"]) # vehicle의 next charging station 값을 바꿔줌
                                min_expect_qcm["waiting_vehicle"] = np.append(min_expect_qcm["waiting_vehicle"], vehicle_id)
                                chargingstation.setParameter(min_expect_qcm["charging_station_id"], "waiting_vehicle", min_expect_qcm["waiting_vehicle"])
                                calcul_temp[curVehicle]+=time.time()
                                calcul_info[curVehicle]+=calcul_temp[curVehicle]
                                time_temp[curVehicle]=simulation.getTime()

                            # 충전소에 도착했지만, 기다려야 할 때, state를 running -> waiting
                            # When you arrive at the charging station but need to wait, change the state to running -> waiting
                            if vehicle.getParameter(vehicle_id, "next_charging_station") and vehicle.getLaneID(vehicle_id)==chargingstation.getLaneID(vehicle.getParameter(vehicle_id, "next_charging_station")):
                                vehicle.setParameter(vehicle_id, "state", "waiting")
                                
                                
                            # 충전소에 도착했고, 충전을 시작했을 때 코드, state를 charging으로 수정 (충전소에 도착했는데, 대기가 없었을 때, running -> charging)
                            # When you arrive at the charging station and start charging, change the code and state to charging (when you arrive at the charging station, but there is no waiting, running -> charging)
                            # if vehicle.getStopState(vehicle_id) == 65:  # 65가 충전소에서 Stop을 의미
                            #     vehicle.setParameter(vehicle_id, "state", "charging")

                            # 자동차가 목적지에 도착했을 때 새로운 목적지 생성
                            # Create a new destination when the car arrives at the destination
                            if vehicle.getRoadID(vehicle_id) == vehicle.getParameter(vehicle_id, "destination"):
                                if vehicle.getParameter(vehicle_id, "next_charging_station") == "":
                                    time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle]
                                    time_list[curVehicle]+=time_temp[curVehicle]
                                    
                                    # 귀환 위치가 아닌 배송지에 도착한 경우에만 info update & 이번 도착지를 dest list에서 삭제 
                                    # Update info only when arriving at the delivery location rather than the return location & delete this destination from the dest list
                                    if finish_info[curVehicle] < destination_number:
                                        time_info[curVehicle][int(finish_info[curVehicle])] = step
                                        finish_info[curVehicle] += 1
                                    nowGoal=vehicle.getParameter(vehicle_id,"destination")
                                    destination_info_g[curVehicle].remove(nowGoal)

                                    if len(destination_info_g[curVehicle])>0: # 아직 가야할 목적지가 남아있는 경우 # If there are still destinations to go
                                        # 남은 목적지 리스트에서 가장 가까운 목적지 검색
                                        # Search for the nearest destination from the list of remaining destinations
                                        calcul_temp[curVehicle]=(-1)*time.time()
                                        minTime=simulation.findRoute(destination_info_g[curVehicle][0], nowGoal).travelTime
                                        nextGoal=destination_info_g[curVehicle][0] # 초기값 지정 # Specify initial value

                                        for g in range(len(destination_info_g[curVehicle])):
                                            if minTime>simulation.findRoute(destination_info_g[curVehicle][g], nowGoal).travelTime:
                                                minTime=simulation.findRoute(destination_info_g[curVehicle][g], nowGoal).travelTime
                                                nextGoal=destination_info_g[curVehicle][g]

                                        # 남은 목적지 중 가장 가까운 곳으로 destination 설정
                                        # Set destination to the closest of the remaining destinations
                                        vehicle.setParameter(vehicle_id, "destination", nextGoal)
                                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                                        calcul_temp[curVehicle]+=time.time()
                                        calcul_info[curVehicle]+=calcul_temp[curVehicle]
                                        time_temp[curVehicle]=simulation.getTime()

                                    else: # 배송 완료
                                        vehicle.remove(vehicle_id)
                                        # print("현재 달성률 {}%".format(sum(finish_info)/(destination_number*vehiclenum)*100))
                                        # print("vehicle_{} 배송 완료, 차량 {}대 중 배송이 남은 차량 {}대".format(curVehicle, vehiclenum, vehicle.getIDCount()))
                                        if vehicle.getIDCount()==0:
                                            dataWrite("Greedy", repeat, vehiclenum, destination_number, charge_info, time_list, calcul_info, Greedy_writer)
                                

                        
                        # vehicle 상태가 waiting이었는데 자기 차례가 이제 됐다면, waiting to charging
                        # If the vehicle status was waiting and it is now your turn, waiting to charging
                        elif vehicle.getParameter(vehicle_id, "state") == "waiting" and vehicle.getStopState(vehicle_id) == 65:
                            charge_temp[curVehicle]=(-1)*simulation.getTime()
                            vehicle.setParameter(vehicle_id, "state", "charging")
                            
                            
                        # vehicle 상태가 Charging이면   # If the vehicle status is Charging
                        elif vehicle.getParameter(vehicle_id, "state") == "charging":
                            battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                            battery_amount += 4
                            vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)

                            # 배터리 양이 최대치로 충전 됐을 때, 충전중인 자동차들 다시 목적지로 출발
                            # When the battery capacity is fully charged, the cars being charged depart again for their destination.
                            if battery_amount >= battery_full_amount:
                                charge_temp[curVehicle]+=simulation.getTime()
                                charge_info[curVehicle]+=charge_temp[curVehicle]
                                vehicle.resume(vehicle_id)
                                vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                                vehicle.setParameter(vehicle_id, "state", "running")
                                
                                for charging_station_dict in charging_station_list:
                                    if charging_station_dict["charging_station_id"] == vehicle.getParameter(vehicle_id, "next_charging_station"):
                                        charging_station_dict_wanted = charging_station_dict
                                        break
                                
                                charging_station_dict_wanted["waiting_vehicle"] = charging_station_dict_wanted["waiting_vehicle"][charging_station_dict_wanted["waiting_vehicle"] != vehicle_id]  # 조건에 맞는 요소만 return하나 봄
                                chargingstation.setParameter(charging_station_dict_wanted["charging_station_id"], "waiting_vehicle", charging_station_dict_wanted["waiting_vehicle"])
                                vehicle.setParameter(vehicle_id, "next_charging_station", "")
                    
                step += 1
                
        elif iterNum==2: # Branch and Bound
            
            
            print("IN BNB CASE")
            for i in range(0, vehiclenum):
                vehicle.add(vehID="vehicle_" + str(car_index), routeID="route_0", typeID='ElectricCar', departPos=0)
                car_index += 1
                
            while simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                for vehicle_id in vehicle.getIDList():
                    # 초기화 코드
                    # vehicle state가 running이 아니면,
                    curVehicle=int(vehicle_id.split('_')[1])
                    if vehicle.getParameter(vehicle_id, "state") == "":
                        vehicle.setParameter(vehicle_id, "actualBatteryCapacity", 1500)
                        vehicle.setParameter(vehicle_id, "state", "ready")
                        time_temp[curVehicle]=simulation.getTime()
                        # TSP
                        for i in range(destination_number+1):
                            if i==0:
                                tot_destination_info[curVehicle][i]=vehicle.getRoadID(vehicle_id)
                                # 출발지 추가 # Add departure point
                            else:
                                tot_destination_info[curVehicle][i]=destination_info[curVehicle][i-1]
                                # 배송지 추가  # Add shipping address

                        min_value = [sys.maxsize]

                        for outer in range(destination_number+1):
                            for inner in range(destination_number+1):
                                if inner==outer:
                                    matrix[curVehicle][outer][inner][1]=tot_destination_info[curVehicle][inner]
                                    continue
                                matrix[curVehicle][outer][inner][0]=simulation.findRoute(tot_destination_info[curVehicle][outer],tot_destination_info[curVehicle][inner]).travelTime
                                matrix[curVehicle][outer][inner][1]=tot_destination_info[curVehicle][inner]
                        calcul_temp[curVehicle]=(-1)*time.time()
                        dfs(0,0,0,[0],destination_number+1, matrix[curVehicle], min_value, shortest_path_b[curVehicle])
                        calcul_temp[curVehicle]+=time.time()
                        calcul_info[curVehicle]+=calcul_temp[curVehicle]
                        

                    elif vehicle.getParameter(vehicle_id, "state") == "ready":
                        vehicle.setParameter(vehicle_id, "state", "running")
                        vehicle.setParameter(vehicle_id, "destination", tot_destination_info[curVehicle][shortest_path_b[curVehicle][0][1]])  # vehicle_1 <- 1 extract, -> vehicle 1은 1 destination info를 선택

                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                        vehicle.setParameter(vehicle_id, "next_charging_station", "")
                        
                        for i in range(destination_number+1):
                            final_path[curVehicle][i]=tot_destination_info[curVehicle][shortest_path_b[curVehicle][0][i]]

                        del final_path[curVehicle][0:2]
                        # 출발지(0번째 인덱스에 위치)랑 위에서 이미 target으로 지정한 1번째 배송 목적지(1번째 인덱스에 위치) 삭제
                        # Delete the origin (located at the 0th index) and the 1st delivery destination (located at the 1st index) already specified as target above.
                        

                    else:
                        # vehicle 상태가 Running이면
                        if vehicle.getParameter(vehicle_id, "state") == "running":
                            battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))  # 현재 값을 가져와서 # Get the current value
                            battery_amount -= 1  # 1 줄이고  #1 Reduce
                            vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)  # 줄인 값을 다시 할당  # Reassign the reduced value
                            
                            # 배터리의 양이 Threshold일때  # When the battery amount is Threshold
                            nowBatt=float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                            needBatt=simulation.findRoute(vehicle.getRoadID(vehicle_id), vehicle.getParameter(vehicle_id, "destination")).travelTime+25

                            if vehicle.getParameter(vehicle_id, "next_charging_station") == "" and nowBatt < needBatt:
                                time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle]
                                time_list[curVehicle]+=time_temp[curVehicle]
                                calcul_temp[curVehicle]=(-1)*time.time()
                                min_expect_time = 100000000
                                min_expect_qcm = search_qcm(charging_station_list, vehicle_id)

                                for charging_station_dict in charging_station_list:
                                    
                                    vehPos=vehicle.getRoadID(vehicle_id)
                                    chstnPos=lane.getEdgeID(chargingstation.getLaneID(charging_station_dict["charging_station_id"]))
                                    needBattToChstn=simulation.findRoute(vehicle.getRoadID(vehicle_id),lane.getEdgeID(chargingstation.getLaneID(charging_station_dict["charging_station_id"]))).travelTime

                                    if (vehPos != chstnPos) and (nowBatt > needBattToChstn):
                                        min_expect_time = simulation.findRoute(vehicle.getRoadID(vehicle_id), lane.getEdgeID(chargingstation.getLaneID(min_expect_qcm["charging_station_id"]))).travelTime

                                        if min_expect_time >= needBattToChstn:
                                            min_expect_qcm = charging_station_dict

                                vehicle.changeTarget(vehicle_id, lane.getEdgeID(chargingstation.getLaneID(min_expect_qcm["charging_station_id"])))
                                vehicle.setChargingStationStop(vehicle_id, min_expect_qcm["charging_station_id"])
                                vehicle.setParameter(vehicle_id, "next_charging_station", min_expect_qcm["charging_station_id"])
                                min_expect_qcm["waiting_vehicle"] = np.append(min_expect_qcm["waiting_vehicle"], vehicle_id)
                                chargingstation.setParameter(min_expect_qcm["charging_station_id"], "waiting_vehicle", min_expect_qcm["waiting_vehicle"])
                                
                                calcul_temp[curVehicle]+=time.time()
                                calcul_info[curVehicle]+=calcul_temp[curVehicle]
                                time_temp[curVehicle]=simulation.getTime() # restart

                            # 충전소에 도착했지만, 기다려야 할 때, state를 running -> waiting
                            # When you arrive at the charging station but need to wait, change the state to running -> waiting
                            if vehicle.getParameter(vehicle_id, "next_charging_station") and vehicle.getLaneID(vehicle_id)==chargingstation.getLaneID(vehicle.getParameter(vehicle_id, "next_charging_station")):
                                vehicle.setParameter(vehicle_id, "state", "waiting")


                            # 자동차가 목적지에 도착했을 때 새로운 목적지 생성
                            # Create a new destination when the car arrives at the destination
                            if vehicle.getRoadID(vehicle_id) == vehicle.getParameter(vehicle_id, "destination"):
                                if vehicle.getParameter(vehicle_id, "next_charging_station") == "":
                                    
                                    time_temp[curVehicle]=simulation.getTime()-time_temp[curVehicle]
                                    time_list[curVehicle]+=time_temp[curVehicle]
                                    
                                    if finish_info[curVehicle] < destination_number:
                                        time_info[curVehicle][int(finish_info[curVehicle])] = step
                                        finish_info[curVehicle] += 1
                                    
                                    if len(final_path[curVehicle])>0: #가야 할 목적지가 남았음 #There are still destinations to go
                                        #heuristic
                                        vehicle.setParameter(vehicle_id, "destination", final_path[curVehicle][0])
                                        del final_path[curVehicle][0]
                                        vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                                        time_temp[curVehicle]=simulation.getTime() # restart

                                    else: #가야 할 목적지가 없음(배송 완료)  #No destination to go (delivery complete)
                                        vehicle.remove(vehicle_id)
                                        if vehicle.getIDCount()==0: #모든 차량 배송 완료  #All vehicles delivered
                                            dataWrite("BnB", repeat, vehiclenum, destination_number, charge_info, time_list, calcul_info, BnB_writer)
                                            

                        # vehicle 상태가 waiting이었는데 자기 차례가 이제 됐다면, waiting to charging
                        # If the vehicle status was waiting and it is now your turn, waiting to charging
                        elif vehicle.getParameter(vehicle_id, "state") == "waiting" and vehicle.getStopState(vehicle_id) == 65:
                            charge_temp[curVehicle]=(-1)*simulation.getTime()
                            vehicle.setParameter(vehicle_id, "state", "charging")
                            
                            
                        # vehicle 상태가 Charging이면 # If the vehicle status is Charging
                        elif vehicle.getParameter(vehicle_id, "state") == "charging":
                            battery_amount = float(vehicle.getParameter(vehicle_id, "actualBatteryCapacity"))
                            battery_amount += 4
                            vehicle.setParameter(vehicle_id, "actualBatteryCapacity", battery_amount)

                            # 배터리 양이 최대치로 충전 됐을 때, 충전중인 자동차들 다시 목적지로 출발
                            # When the battery capacity is fully charged, the cars being charged depart again for their destination.
                            if battery_amount >= battery_full_amount:
                                charge_temp[curVehicle]+=simulation.getTime()
                                charge_info[curVehicle]+=charge_temp[curVehicle]
                                vehicle.changeTarget(vehicle_id, vehicle.getParameter(vehicle_id, "destination"))
                                vehicle.setParameter(vehicle_id, "state", "running")
                                vehicle.setParameter(vehicle_id, "next_charging_station", "")
                                vehicle.resume(vehicle_id)

                step += 1


    traci.close()
    sys.stdout.flush()
    

if __name__ == "__main__":

    options = get_options() # 通过 get_options() 函数获取命令行参数（比如是否使用图形界面 nogui 选项等）。
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')


    ML_file=open('MLSAINT_more_perfo.csv', 'w', newline='') #创建文件记录与 ML 算法相关的性能数据
    Greedy_file=open('Greedy_perfo_data.csv', 'w', newline='') # 记录贪婪算法的性能数据
    BnB_file=open('BnB_perfo_data.csv', 'w', newline='') # 分支定界算法的性能数据。

    #创建文件内的内容标题行
    entry = [['Algorithm', 'laps', 'Vehicle number', 'Destination number', 'Charging Time', 'Running Time', 'Calculation Time', 'Average Link Delay', 'Average Max Delay']]

    #创建三个写入器
    ML_writer = csv.writer(ML_file)
    Greedy_writer = csv.writer(Greedy_file)
    BnB_writer = csv.writer(BnB_file)

    #将标题行分别写入下面三个文件
    ML_writer.writerows(entry)
    Greedy_writer.writerows(entry)
    BnB_writer.writerows(entry)
   
            
    # Destination and Vehicle Number diff
    for d in range(0, 10, 1): # 遍历 0 到 9，共 10 次，d 用来控制目的地的数量。
        for repeat in range(1, 11, 1):
            vNum=50 # 每次仿真的车辆数量设定为50。
            if d==0:
                dNum=1
            else:
                dNum=5*d

            #  dNum: 1, 5, 10, 15, 20, 25, 30, 35, 40, 45
            # repeat: 1, 2,  3,  4,  5,  6,  7,  8,  9, 10
                
            try:
                #加载配置文件
                #traci.start([sumoBinary, "-c", "demo_30.sumocfg"])
                traci.start([sumoBinary, "-c", "Gangnam4.sumocfg"]) # 启动SUMO仿真，使用选定的二进制文件sumoBinary和配置文件Gangnam4.sumocfg。traci是SUMO的Python接口，用于控制SUMO仿真过程。
                start = time.time()# 记录仿真开始时间。
                print(f"开始运行run函数！！")
                print(f"vFum:{vNum}, dNum:{dNum}, repeat:{repeat}")
                run(vNum, dNum, ML_file, Greedy_file, BnB_file, repeat) # 调用run函数，传递 (车辆数量vNum, 目的地数量dNum, CSV写入器SAINT_writer记录结果, 重复次数repeat), 运行仿真并记录结果。
                
            except Exception:  # 捕获并忽略在仿真过程中发生的任何异常，以确保程序即使发生错误也不会崩溃。
                print("有错误！")
                pass
