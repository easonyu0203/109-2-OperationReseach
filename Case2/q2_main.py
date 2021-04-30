from os import path
import os
import pandas as pd
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from gurobipy import quicksum
import math

from q2_ulti import *

work_dir = os.getcwd()
data_path = path.join(work_dir, 'data')

data_file_list = [path.join(data_path, f) for f in os.listdir(data_path) if '~$' not in f]
data_file_list.sort()

for data_file in data_file_list:
    print()
    instance_name = data_file[-7:-5]
    print(f'working with {instance_name}')
    #get parameters
    (
        N, M, K, V,
        ContainerCap, ContainerCost,
        Demand_ij,
        Init_i,
        BuyCost_i,
        HoldCost_i,
        Transit_ij,
        ShipFixedCost_k,
        ShipVarCost_ik,
        CBM_i,
        LostSaleCost_i,
        BackOrderCost_i,
        BackOrderProb_i,
        VendorFixedCost_v,
        MinOrder_i,
        ConflictPair_alpha,
        ProductVendor_i,
        M_big,
    ) = get_data(data_file)

    #init model
    model = gb.Model(instance_name)

    #decision variable
    x = model.addVars(N, M, K, vtype=GRB.INTEGER,name='x')
    Abin = model.addVars(M, K, vtype=GRB.BINARY, name='Abin')
    ContainerCnt = model.addVars(M, vtype=GRB.INTEGER, name='ContainerCnt')
    StockLevel = model.addVars(N, M, vtype=GRB.CONTINUOUS, name='StockLevel')
    Shortage = model.addVars(N, M, vtype=GRB.CONTINUOUS, name='Shortage')
    Bbin = model.addVars(N, M, vtype=GRB.BINARY, name='Bbin')
    Cbin = model.addVars(N, M, vtype=GRB.BINARY, name='Cbin')
    Dbin = model.addVars(V, M, vtype=GRB.BINARY, name='Dbin')

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

    TotalShipFixedCost = gb.LinExpr(
        quicksum(
            quicksum(
                Abin[j,k] * ShipFixedCost_k[k]
            for k in range(K))
        for j in range(M))
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

    TotalVendorFixedCost = gb.LinExpr(
        quicksum(
            quicksum(
                Dbin[v,j] * VendorFixedCost_v[v]
            for v in range(V))
        for j in range(M))
    )

    #set obj function
    model.setObjective(
        TotalPurchaseCost +
        TotalShipFixedCost + 
        TotalShipVarCost + 
        TotalHoldingCost + 
        TotalContainerCost + 
        TotalBackOrderCost + 
        TotalLostSaleCost +
        TotalVendorFixedCost
    )

    #Contrain

    #let Abin to behave correctly
    _ = model.addConstrs(
        quicksum(x[i,j,k] for i in range(N)) / M_big <= Abin[j,k]
    for j in range(M)
    for k in range(K))

    #let ContainerCnt to behave correctly
    _ = model.addConstrs(
        VolumeInOceanExpr_j[j] / ContainerCap <= ContainerCnt[j]
    for j in range(M))

    #let Stocklevle & Shortage behave correctly
    _ = model.addConstrs(StockLevel[i,0] - Shortage[i,0] == Init_i[i] - Demand_ij[i][0] for i in range(N))
    _ = model.addConstrs(StockLevel[i,1] - Shortage[i,1] == StockLevel[i,0] + x[i,0,0] + Transit_ij[i][1] - Demand_ij[i][1] - Shortage[i,0] * BackOrderProb_i[i] for i in range(N))
    _ = model.addConstrs(StockLevel[i,2] - Shortage[i,2] == StockLevel[i,1] + x[i,1,0] + x[i,0,1] + Transit_ij[i][2] - Demand_ij[i][2] - Shortage[i,1] * BackOrderProb_i[i] for i in range(N))
    _ = model.addConstrs(StockLevel[i,j] - Shortage[i,j] == StockLevel[i,j-1] + x[i,j-1,0] + x[i,j-2,1] + x[i,j-3,2] - Demand_ij[i][j] - Shortage[i,j-1] * BackOrderProb_i[i] for i in range(N) for j in range(3,M))

    _ = model.addConstrs(StockLevel[i,j] <= M_big * (1-Bbin[i,j]) for i in range(N) for j in range(M))
    _ = model.addConstrs(Shortage[i,j] <= M_big * Bbin[i,j] for i in range(N) for j in range(M))

    #let Cbin behave correctly
    _ = model.addConstrs(
        quicksum(x[i,j,k] for k in range(K)) / M_big <= Cbin[i,j]
    for i in range(N)
    for j in range(M))

    #let Dbin behave correctly
    _ = model.addConstrs(
        quicksum(
            quicksum(x[i,j,k] for k in range(K))
        for i in range(N) if ProductVendor_i[i] == v) / M_big <= Dbin[v,j]
    for v in range(V)
    for j in range(M))

    #Minumin order Bound
    _ = model.addConstrs(
        Cbin[i,j] * MinOrder_i[i] <= quicksum(x[i,j,k] for k in range(K))
    for i in range(N)
    for j in range(M))

    #conflict
    _ = model.addConstrs(
        Cbin[a,j] + Cbin[b,j] <= 1
    for j in range(M)
    for a, b in ConflictPair_alpha)

    #optimize
    model.optimize()

    #result
    solution_file_path = path.join(os.getcwd(), 'solution', f'sol_{instance_name}.xlsx')
    with pd.ExcelWriter(solution_file_path, engine='xlsxwriter') as writer:

        df = pd.DataFrame([[int(x[i,j,0].x) for j in range(M)] for i in range(N)], columns=range(M))
        df.to_excel(writer, sheet_name='Express delivery')
        
        df = pd.DataFrame([[int(x[i,j,1].x) for j in range(M)] for i in range(N)], columns=range(M))
        df.to_excel(writer, sheet_name='Air frieght')
        
        df = pd.DataFrame([[int(x[i,j,2].x) for j in range(M)] for i in range(N)], columns=range(M))
        df.to_excel(writer, sheet_name='Ocean frieght')
        
        df = pd.DataFrame([[StockLevel[i,j].x for j in range(M)] for i in range(N)], columns=range(M))
        df.to_excel(writer, sheet_name='StockLevel')
        
        df = pd.DataFrame([[Shortage[i,j].x for j in range(M)] for i in range(N)], columns=range(M))
        df.to_excel(writer, sheet_name='Shortage')

        result_df = pd.DataFrame(columns=[
            'TotalPurchaseCost',
            'TotalShipFixedCost',
            'TotalShipVarCost',
            'TotalHoldingCost',
            'TotalContainerCost',
            'TotalBackOrderCost',
            'TotalLostSaleCost',
            'TotalVendorFixedCost',
            'objective-value',
        ])

        result_df.loc[0] = [
            str(round(TotalPurchaseCost.getValue(), 2)),
            str(round(TotalShipFixedCost.getValue(), 2)),
            str(round(TotalShipVarCost.getValue(), 2)),
            str(round(TotalHoldingCost.getValue(), 2)),
            str(round(TotalContainerCost.getValue(), 2)),
            str(round(TotalBackOrderCost.getValue(), 2)),
            str(round(TotalLostSaleCost.getValue(), 2)),
            str(round(TotalVendorFixedCost.getValue(), 2)),
            str(round(model.ObjVal, 2)),
        ]

        result_df.to_excel(writer, sheet_name='result')
