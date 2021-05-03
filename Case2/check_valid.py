import time
import pandas as pd
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
from algorithm_module import heuristic_algorithm
from q2_ulti import get_data
import argparse

def check_feasibility_and_calculate_objective_value(file_path: str, order: list) -> tuple:
    """ return typle -> (is feasible, obj value) """
    #work with indice start from 0
    order = np.array(order)
    order = order[1:,1:,1:]
    
    is_valid = False
    #check feasible or not
    (
        N, M, K, V,
        ContainerCap, ContainerCost,
        Demand_it,
        Init_i,
        BuyCost_i,
        HoldCost_i,
        Transit_it,
        ShipFixedCost_j,
        ShipVarCost_ij,
        CBM_i,
        LostSaleCost_i,
        BackOrderCost_i,
        BackOrderProb_i,
        VendorFixedCost_v,
        MinOrder_i,
        ConflictPair_alpha,
        ProductVendor_i,
        M_big,
    ) = get_data(file_path)

    #min order bound
    month_order = np.sum(order, axis=1)
    surpus_min = ((month_order.transpose() - MinOrder_i).transpose() >= 0)
    is_zero = (month_order == 0)
    if np.any((np.logical_not(surpus_min | is_zero))):
        print('min order bound is not satify')
        print('list of not satisfy')
        for i in range(N):
            for j in range(M):
                if month_order[i,j] == 0: continue
                if month_order[i,j] < MinOrder_i[i]:
                    print(f'order product {i+1} month {j+1}')
        is_valid = False
    
    #conflict
    conflict_order = []
    have_buy = np.sum(order, axis=1) > 0
    for j, month in enumerate(have_buy.transpose()):
        for p1, p2 in ConflictPair_alpha:
            if month[p1] and month[p2]:
                conflict_order.append(f'month {j+1}: {p1}, {p2}')
    if len(conflict_order) != 0:
        print('conflict happend')
        print('\n'.join(conflict_order))
        is_valid = False

    is_valid = True

    Inven = np.zeros((N,M))
    for i in range(N):
        Inven[i,0] = Init_i[i] - Demand_it[i,0]
        Inven[i,1] = max(0, Inven[i,0]) + order[i,0,0] + Transit_it[i,1] - Demand_it[i,1] + min(0, Inven[i,0]) * BackOrderProb_i[i]
        Inven[i,2] = max(0, Inven[i,1]) + order[i,0,1] + order[i,1,0] + Transit_it[i,2] - Demand_it[i,2] + min(0, Inven[i,1]) * BackOrderProb_i[i]
        for t in range(3,M):
            Inven[i,t] = max(0, Inven[i,t-1]) + order[i,0,t-1] + order[i,1,t-2] + order[i,2,t-3] - Demand_it[i,t] + min(0, Inven[i,t-1]) * BackOrderProb_i[i]

    #obj value
    purchase_cost = np.sum(order.transpose((1,2,0)) * BuyCost_i)
    shipFixed_cost = np.sum(np.sum(np.sum(order, axis=0) > 0, axis=1) * ShipFixedCost_j)
    shipVar_cost = np.sum(np.sum(order, axis=2) * ShipVarCost_ij)
    container_cost = sum([ i // ContainerCap if i % ContainerCap == 0 else i // ContainerCap + 1 for i in np.sum((order[:,2,:].transpose() * CBM_i).transpose(),  axis=0)]) * ContainerCost
    vendorFixed_cost = sum([np.sum(np.sum(order[[i for i in range(N) if v == ProductVendor_i[i]],:,:], axis=(0,1)) > 0) * VendorFixedCost_v[v] for v in range(V)])
    holding_cost = np.sum((Inven.transpose() * HoldCost_i).transpose()[Inven > 0])
    backOrder_cost = np.sum((-Inven.transpose() * BackOrderCost_i * BackOrderProb_i).transpose()[Inven < 0])
    lostSale_cost = np.sum((-Inven.transpose() * LostSaleCost_i * (1-BackOrderProb_i)).transpose()[Inven < 0])
    obj_val = purchase_cost + shipFixed_cost + shipVar_cost + container_cost + vendorFixed_cost + holding_cost + backOrder_cost + lostSale_cost
    
    print(f'TotalPurchaseCost    :{purchase_cost:.2f}')
    print(f'TotalShipFixedCost   :{shipFixed_cost:.2f}')
    print(f'TotalShipVarCost     :{shipVar_cost:.2f}')
    print(f'TotalHoldingCost     :{holding_cost:.2f}')
    print(f'TotalContainerCost   :{container_cost:.2f}')
    print(f'TotalBackOrderCost   :{backOrder_cost:.2f}')
    print(f'TotalLostSaleCost    :{lostSale_cost:.2f}')
    print(f'TotalVendorFixedCost :{vendorFixed_cost:.2f}')
    print(f'objective-value      :{obj_val:.2f}')

    return (is_valid, obj_val)
    


    

