#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:04:45 2019

@author: jeff
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table_experiments as dt

from dash.dependencies import Input, Output
import dash_table



df = pd.read_csv('/home/jeff/Documents/workspace/Machine_learning_semantique/frontEnd/dashboard/myFileDP_add.csv', error_bad_lines=False)

df[' index'] = range(1, len(df) + 1)

app = dash.Dash(__name__)

PAGE_SIZE = 5

app.layout = dash_table.DataTable(
    id='datatable-paging',
    columns=[
        {"name": i, "id": i} for i in sorted(df.columns)
    ],
    pagination_settings={
        'current_page': 0,
        'page_size': PAGE_SIZE
    },
    pagination_mode='be'
)


@app.callback(
    Output('datatable-paging', 'data'),
    [Input('datatable-paging', 'pagination_settings')])
def update_table(pagination_settings):
    return df.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ].to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)