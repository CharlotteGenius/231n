#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:12:18 2020

@author: xuanchenxiang
"""

# *args is used to send a non-keyworded variable length argument list to the function.
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)
        
test_var_args('yasoob', 'python', 'eggs', 'test')

arg = ('python', 'eggs', 'test')
print(*arg)


# **kwargs allows you to pass keyworded variable length of arguments to a function.
# You should use **kwargs if you want to handle named arguments in a function. 
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
    print(kwargs)

greet_me(name="yasoob", test = 't')



# how to call a function using *args and **kwargs
def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

# first with *args
args = ("two", 3, 5)
test_args_kwargs(*args)


# now with **kwargs:
kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)


# if you want to use all three of these in functions then the order is
# some_func(fargs, *args, **kwargs)










