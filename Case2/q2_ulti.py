from os import path
import os
import pandas as pd
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from gurobipy import quicksum
import math


def get_data(data_file):
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

    return(
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
    )

