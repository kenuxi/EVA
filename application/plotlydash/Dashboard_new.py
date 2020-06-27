import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from config import iris_config, external_stylesheets
from abc import abstractmethod, ABC
from .assets.layout import html_layout

import dash_daq as daq
import pandas as pd
from visualisation_methods import VisualizationPlotly
from statistics_methods import DataStatistics

from typing import List, Dict


class Dashboard(ABC):
    def __init__(self, server: flask.Flask, stylesheets: List[str], prefix: str = '/dash'):
        self.server = server
        self.stylesheets = stylesheets
        self.prefix = prefix
        self.dash_app = None

    @abstractmethod
    def create_dashboard(self, data_dict: Dict[str, str]):
        pass

    @abstractmethod
    def init_callbacks(self, target_column):
        pass


class FileDashboard(Dashboard):
    def __init__(self, server, stylsheets, prefix):
        super().__init__(server, stylsheets, prefix)
        self.data = None

    def create_dashboard(self, data_dict: Dict[str, str]):
        data_file_name = data_dict['location']
        dim_red_methods = data_dict['algorithms']
        stats = DataStatistics()
        stats.load_data(file_name=data_file_name)
        self.data = stats.pandas_data_frame


class DimReductionPlot(ABC):
    def __init__(self, inputs, ds):
        self.ds = ds
        self.graph = self._get_graph(inputs)
        self.dropdowns = self._getdropdowns(inputs)
        self.reduced_data = self._reduce_data(inputs)

    @abstractmethod
    def _get_graph(self, inputs):
        pass

    @abstractmethod
    def _get_dropdowns(self, inputs):
        pass

    @abstractmethod
    def _reduce_data(self, inputs):
        pass








