# -*- coding: utf-8 -*-

class PostTestProcess(object):

    def __init__(self, **kwargs):
        pass

    def __call__(self, test_res):
        raise NotImplementedError
