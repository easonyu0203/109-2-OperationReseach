from os import path
import os
import pandas as pd
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from gurobipy import quicksum
import math
from check_valid import check_feasibility_and_calculate_objective_value

def heuristic_algorithm(file_path):
    
    '''
    1. Write your heuristic algorithm here.
    2. We would call this function in CS2_graded_program.py to evaluate your algorithm.
    3. Please do not change the function name and the file name.
    4. The parameter is file path of data file.
    5. You need to return the order plan in list in order.
            order[i][j][t] is amount of product i ordered in the beginning of month t with shipping method j.
            i = 1, ..., |number of product|, j = 1, 2, 3, and t = 1, ..., 26.
       Note that the indice of list need to be in order, that is , return order[i][j][t] rather than order[t][j][i]
    6. You only need to submit this algorithm_module.py.
    '''
    
    #read data parameters
    data_file = pd.ExcelFile(file_path)
    Demand_df = pd.read_excel(data_file, sheet_name='Demand', index_col='Product')
    Init_df = pd.read_excel(data_file, sheet_name='Initial inventory', index_col='Product')
    ShippingCost_df = pd.read_excel(data_file, sheet_name='Shipping cost', index_col='Product')[['Express delivery', 'Air freight']]
    InTransit_df = pd.read_excel(data_file, sheet_name='In-transit', index_col='Product')
    CBM_df = pd.read_excel(data_file, sheet_name='Size', index_col='Product')
    PriceAndCost_df = pd.read_excel(data_file, sheet_name='Price and cost', index_col='Product')
    Shortage_df = pd.read_excel(data_file, sheet_name='Shortage', index_col='Product')
    VendorProduct_df = pd.read_excel(data_file, sheet_name='Vendor-Product', index_col='Product') - 1
    VendorCost_df = pd.read_excel(data_file, sheet_name='Vendor cost', index_col='Vendor')
    Bound_df = pd.read_excel(data_file, sheet_name='Bounds', index_col='Product')
    Conflict_df = pd.read_excel(data_file, sheet_name='Conflict', index_col='Conflict') - 1

    N, M = Demand_df.shape
    K = 3 #(express delivery, air freight, Ocean freight)
    V = VendorCost_df.shape[0]
    # ContainerCap = 30 #Case 1 data
    # ContainerCost = 2750 #Case 1 data
    ContainerCap = 0.5 #Case 1 data
    ContainerCost = 1500 #Case 1 data

    Demand_ij = Demand_df.to_numpy()
    Init_i = Init_df.to_numpy().squeeze()
    BuyCost_i = PriceAndCost_df['Purchasing cost'].to_numpy()
    HoldCost_i = PriceAndCost_df['Holding cost'].to_numpy()
    Transit_ij = InTransit_df.to_numpy()
    ShipFixedCost_k = np.array([100, 80, 50]) #Case 1 data
    ShipVarCost_ik = np.concatenate(( ShippingCost_df.to_numpy(), np.zeros((N,1)) ), axis=1)
    CBM_i = CBM_df.to_numpy().squeeze()
    LostSaleCost_i = Shortage_df['Lost sales'].to_numpy()
    BackOrderCost_i = Shortage_df['Backorder'].to_numpy()
    BackOrderProb_i = Shortage_df['Backorder percentage'].to_numpy()
    VendorFixedCost_v = VendorCost_df.to_numpy().squeeze()
    MinOrder_i = Bound_df.to_numpy().squeeze()
    ConflictPair_alpha = Conflict_df.to_numpy()
    ProductVendor_i = VendorProduct_df.to_numpy().squeeze()
    M_big = np.sum(Demand_ij) + sum(Init_i)

    
    #model
    model = gb.Model()

    #set gurobi param
    model.Params.LogToConsole = 0
    model.Params.MIPGap = (1)/100
    

    #decision variable
    x = model.addVars(N, M, K, vtype=GRB.INTEGER,name='x')
    ContainerCnt = model.addVars(M, vtype=GRB.INTEGER, name='ContainerCnt')
    StockLevel = model.addVars(N, M, vtype=GRB.CONTINUOUS, name='StockLevel')
    Shortage = model.addVars(N, M, vtype=GRB.CONTINUOUS, name='Shortage')
    Bbin = model.addVars(N, M, vtype=GRB.BINARY, name='Bbin')    
    

    #Expr
    VolumeInOceanExpr_j = [
    gb.LinExpr(
        quicksum(x[i,j,2] * CBM_i[i] for i in range(N))
    )    
    for j in range(M)]

    BackOrderCntExpr_ij = [
        [
            gb.LinExpr( Shortage[i,j] * BackOrderProb_i[i] )
        for j in range(M)]
    for i in range(N)]

    LostSaleCntExpr_ij = [
        [
            gb.LinExpr( Shortage[i,j] * (1 - BackOrderProb_i[i]) )
        for j in range(M)]
    for i in range(N)]


    #obj function
    TotalPurchaseCost = gb.LinExpr(
        quicksum(
            quicksum(
                quicksum(
                    x[i,j,k] * BuyCost_i[i]
                for i in range(N))
            for j in range(M))
        for k in range(K))
    )

    TotalShipVarCost = gb.LinExpr(
        quicksum(
            quicksum(
                quicksum(
                    x[i,j,k] * ShipVarCost_ik[i,k]
                for k in range(K))
            for j in range(M))
        for i in range(N))
    )

    TotalHoldingCost = gb.LinExpr(
        quicksum(
            quicksum(
                StockLevel[i,j] * HoldCost_i[i]
            for i in range(N))
        for j in range(M))
    )

    TotalContainerCost = gb.LinExpr(
        quicksum(ContainerCnt[j] for j in range(M)) * ContainerCost
    )

    TotalBackOrderCost = gb.LinExpr(
        quicksum(
            quicksum(
                BackOrderCntExpr_ij[i][j] * BackOrderCost_i[i]
            for i in range(N))
        for j in range(M))
    )

    TotalLostSaleCost = gb.LinExpr(
        quicksum(
            quicksum(
                LostSaleCntExpr_ij[i][j] * LostSaleCost_i[i]
            for i in range(N))
        for j in range(M))
    )


    #set obj function
    model.setObjective(
        TotalPurchaseCost +
        TotalShipVarCost + 
        TotalHoldingCost + 
        TotalContainerCost + 
        TotalBackOrderCost + 
        TotalLostSaleCost
    )


    #Contrain
    #let ContainerCnt to behave correctly
    model.addConstrs(
        VolumeInOceanExpr_j[j] / ContainerCap <= ContainerCnt[j]
    for j in range(M))

    #let Stocklevle & Shortage behave correctly
    model.addConstrs(StockLevel[i,0] - Shortage[i,0] == Init_i[i] - Demand_ij[i][0] for i in range(N))
    model.addConstrs(StockLevel[i,1] - Shortage[i,1] == StockLevel[i,0] + x[i,0,0] + Transit_ij[i][1] - Demand_ij[i][1] - Shortage[i,0] * BackOrderProb_i[i] for i in range(N))
    model.addConstrs(StockLevel[i,2] - Shortage[i,2] == StockLevel[i,1] + x[i,1,0] + x[i,0,1] + Transit_ij[i][2] - Demand_ij[i][2] - Shortage[i,1] * BackOrderProb_i[i] for i in range(N))
    model.addConstrs(StockLevel[i,j] - Shortage[i,j] == StockLevel[i,j-1] + x[i,j-1,0] + x[i,j-2,1] + x[i,j-3,2] - Demand_ij[i][j] - Shortage[i,j-1] * BackOrderProb_i[i] for i in range(N) for j in range(3,M))
    model.addConstrs(StockLevel[i,j] <= M_big * (1-Bbin[i,j]) for i in range(N) for j in range(M))
    model.addConstrs(Shortage[i,j] <= M_big * Bbin[i,j] for i in range(N) for j in range(M))

    #avoud conflict
    p1, p2 = zip(*ConflictPair_alpha)
    _ = model.addConstrs(
        x[i,j,k] == 0
        for i in p1
        for j in range(0, M, 2) #even
        for k in range(K)
    )

    _ = model.addConstrs(
        x[i,j,k] == 0
        for i in p2
        for j in range(1, M, 2) #even
        for k in range(K)
    )

    
    #optimize
    model.optimize()


    #order
    order = np.zeros((N,K,M), dtype=np.float32)
    for i in range(N):
        for j in range(K):
            for t in range(M):
                order[i,j,t] = x[i,t,j].x

    #post order
    month_order = np.sum(order, axis=1)
    surpus_min = ((month_order.transpose() - MinOrder_i).transpose() >= 0)
    is_zero = (month_order == 0)
    not_valid_min_bound = np.logical_not(surpus_min | is_zero)
    not_valid_min_bound_indices =  not_valid_min_bound.nonzero()
    for i, t in zip(*not_valid_min_bound_indices):
        if sum(order[i,j,t] for j in range(K)) >= MinOrder_i[i]:
            continue
        if i in ConflictPair_alpha[0] or i in ConflictPair_alpha[1]:
            if order[i,0,t] != 0 and order[i,1,t] != 0:
                if month_order[i,t+2] != 0:
                    order[i,0,t+2] += order[i,2,t]
                    order[i,2,t] = 0
        else:
            if order[i,0,t] != 0: #can't post
                continue
            if month_order[i,t+1] != 0: #under min bound or over min bound
                order[i,0,t+1] += order[i,1,t]
                order[i,1,t+1] += order[i,2,t]
                order[i,1,t], order[i,2,t] = 0, 0
            else:
                if order[i,1,t] != 0: #can't post
                    continue
                if month_order[i,t+2] != 0:
                    order[i,0,t+2] += order[i,2,t]
                    order[i,2,t] = 0

    #pre order
    month_order = np.sum(order, axis=1)
    surpus_min = ((month_order.transpose() - MinOrder_i).transpose() >= 0)
    is_zero = (month_order == 0)
    not_valid_min_bound = np.logical_not(surpus_min | is_zero)
    not_valid_min_bound_indices = (np.flip(p, axis=0) for p in not_valid_min_bound.nonzero())
    for i, t in zip(*not_valid_min_bound_indices): #from back to begin
        if sum(order[i,j,t] for j in range(K)) >= MinOrder_i[i]:
            continue
        if i in ConflictPair_alpha[0] or i in ConflictPair_alpha[1]:
            if t - 2 >= 0 and month_order[i,t-2] != 0:
                order[i,2,t-2] += order[i,0,t] + order[i,1,t] + order[i,2,t]
                order[i,0,t], order[i,1,t], order[i,2,t] = 0,0,0
        else:
            if t - 1 >= 0 and month_order[i,t-1] != 0:
                order[i,1,t-1] += order[i,0,t]
                order[i,2,t-1] += order[i,1,t]
                order[i,2,t-1] += order[i,2,t]
                order[i,0,t], order[i,1,t], order[i,2,t] = 0,0,0
            elif t - 2 >= 0 and month_order[i,t-2] != 0:
                order[i,2,t-2] += order[i,0,t] + order[i,1,t] + order[i,2,t]
                order[i,0,t], order[i,1,t], order[i,2,t] = 0,0,0
    
    #move some order to front
    month_order = np.sum(order, axis=1)
    surpus_min = ((month_order.transpose() - MinOrder_i).transpose() >= 0)
    is_zero = (month_order == 0)
    not_valid_min_bound = np.logical_not(surpus_min | is_zero)
    not_valid_min_bound_indices =  not_valid_min_bound.nonzero()
    for i, t in zip(*not_valid_min_bound_indices):
        need = MinOrder_i[i] - sum(order[i,j,t] for j in range(K))
        target_t = t
        while need > 0:
            target_t += 1
            if target_t < M:
                can_offer = sum(order[i,j,target_t] for j in range(K)) - MinOrder_i[i]
                if can_offer > 0:
                    if target_t == t + 1:
                        #use 0,1,2 delivery
                        if need >= can_offer: #take all
                            if order[i,0,target_t] >= can_offer:
                                order[i,0,target_t], order[i,1,t], can_offer = order[i,0,target_t] - can_offer, order[i,1,t] + can_offer, 0
                            elif order[i,0,target_t] + order[i,1,target_t] >= can_offer:
                                order[i,0,target_t], order[i,1,t], can_offer = 0, order[i,1,t] + order[i,0,target_t], can_offer - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], can_offer = order[i,1,target_t] - can_offer, order[i,2,t] + can_offer, 0
                            else:
                                order[i,0,target_t], order[i,1,t], can_offer = 0, order[i,1,t] + order[i,0,target_t], can_offer - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], can_offer = 0, order[i,2,t] + order[i,1,target_t], can_offer - order[i,1,target_t]
                                order[i,2,target_t], order[i,2,t], can_offer = order[i,2,target_t] - can_offer, order[i,2,t] + can_offer, 0

                        else:
                            if order[i,0,target_t] >= need:
                                order[i,0,target_t], order[i,1,t], need = order[i,0,target_t] - need, order[i,1,t] + need, 0
                            elif order[i,0,target_t] + order[i,1,target_t] >= need:
                                order[i,0,target_t], order[i,1,t], need = 0, order[i,1,t] + order[i,0,target_t], need - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], need = order[i,1,target_t] - need, order[i,2,t] + need, 0
                            else: 
                                order[i,0,target_t], order[i,1,t], need = 0, order[i,1,t] + order[i,0,target_t], need - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], need = 0, order[i,2,t] + order[i,1,target_t], need - order[i,1,target_t]
                                order[i,2,target_t], order[i,2,t], need = order[i,2,target_t] - need, order[i,2,t] + need, 0
                    else:
                        if need >= can_offer: #take all
                            if order[i,0,target_t] >= can_offer:
                                order[i,0,target_t], order[i,2,t], can_offer = order[i,0,target_t] - can_offer, order[i,2,t] + can_offer, 0
                            elif order[i,0,target_t] + order[i,1,target_t] >= can_offer:
                                order[i,0,target_t], order[i,2,t], can_offer = 0, order[i,2,t] + order[i,0,target_t], can_offer - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], can_offer = order[i,1,target_t] - can_offer, order[i,2,t] + can_offer, 0
                            else:
                                order[i,0,target_t], order[i,2,t], can_offer = 0, order[i,2,t] + order[i,0,target_t], can_offer - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], can_offer = 0, order[i,2,t] + order[i,1,target_t], can_offer - order[i,1,target_t]
                                order[i,2,target_t], order[i,2,t], can_offer = order[i,2,target_t] - can_offer, order[i,2,t] + can_offer, 0

                        else:
                            if order[i,0,target_t] >= need:
                                order[i,0,target_t], order[i,2,t], need = order[i,0,target_t] - need, order[i,2,t] + need, 0
                            elif order[i,0,target_t] + order[i,1,target_t] >= need:
                                order[i,0,target_t], order[i,2,t], need = 0, order[i,2,t] + order[i,0,target_t], need - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], need = order[i,1,target_t] - need, order[i,2,t] + need, 0
                            else: 
                                order[i,0,target_t], order[i,2,t], need = 0, order[i,2,t] + order[i,0,target_t], need - order[i,0,target_t]
                                order[i,1,target_t], order[i,2,t], need = 0, order[i,2,t] + order[i,1,target_t], need - order[i,1,target_t]
                                order[i,2,target_t], order[i,2,t], need = order[i,2,target_t] - need, order[i,2,t] + need, 0
            else:
                order[i,2,t] += need
                need = 0
    
    out = np.zeros((N+1,K+1,M+1), dtype=np.float32)
    out[1:,1:,1:] = order
    return out.tolist()