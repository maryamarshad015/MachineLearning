# -*- coding: utf-8 -*-
"""Machine Learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HWMX9caCaOkScCJlsop7wyHCU_w-yPDz

Lab 01: Introduction to Python for Machine Learning

Lab Overview

In this lab, we will cover the following topics:

1.A brief review of Python for machine learning.

2.Setting up and using Google Colab.

3.Learning essential Python concepts that are foundational for machine learning.

4.Performing basic Python coding tasks.

5.Assigning a homework task for additional practice.

Objective

By the end of this lab, students will be able to:

1.Set up and run Python code in Google Colab.

2.Understand basic Python concepts, including data types, loops, functions, and libraries like NumPy and Pandas.

3.Write simple Python scripts essential for machine learning.

Task 1: Python Basics Review

Create a Python script that:

1.Defines two variables (one integer, one string).
"""

number=45
word = 'welcome'

"""2.Uses an if-else statement to print a message depending on the value of the integer."""

if number > 32:
  print(f'number:{number}')
else:
  print("False")

"""3.Creates a list of five elements and prints the first three.


"""

list_num = [2,3,5,6,7]
#slicing
print(list_num[:3])

#for loop in list comprehension
[print(num) for num in list_num[:3]]

"""4.Defines a function that multiplies two numbers and prints the result."""

def multiply(num1,num2):
  return num1*num2

print(multiply(2,3))

"""Task 2: NumPy and Pandas"""

import numpy as np
import pandas as pd

"""1.Create a NumPy array of numbers from 1 to 10 and calculate the mean."""

np_arr = np.array(range(1,10+1))
print(np_arr.mean())

"""2.Create a Pandas DataFrame using a dictionary with at least two columns (e.g., Name and Score) and print the first two rows."""

score_dict = {'Name':['Nick','Donald','John'],'Score':[34,54,66]}
df = pd.DataFrame(score_dict)
df.iloc[:2]

"""Part 5: Homework Assignment

Homework Task 1: Python Functions and Loops

Write a Python program that:

1.Accepts user input for three numbers.

2.Calculates and prints the average of these numbers using a function.

"""

n1 = int(input("Num1:"))
n2 = int(input("Num2:"))
n3 = int(input("Num3:"))

np_arr = np.array([n1,n2,n3])

#printing avg
print(f'Average: {round(np_arr.mean(),2)}')

"""3.Uses a loop to display numbers from 1 to 10, but only prints even numbers."""

print([num for num in range(10+1) if num%2 == 0])

"""Homework Task 2: NumPy and Data Analysis
Write a Python script in Colab that:

1.Creates a NumPy array of random numbers (size 10) and calculates the standard deviation.
"""

import random

array = np.array([random.randint(1,100) for num in range(10)])
print(f'Standard Deviation: {round(array.std(),2)}')

"""2.Loads a dataset into a Pandas DataFrame, prints the column names, and displays the summary statistics for numerical columns."""

dataset = pd.read_csv('CrimesOnWomenIndia.csv')
dataframe = pd.DataFrame(dataset)
dataframe.head()

dataframe.describe()

"""Task: 1
  1. Create a Python script that accepts a user's first and last name as input.
  2. Combine the names and print a welcome message, e.g., "Welcome, John Doe!"

"""

f_name = input("First Name:")
l_name = input("Last Name:")

print(f"Welcome {' '.join([f_name,l_name])}!")

"""3. Create a list of five cities and use a loop to print each city in uppercase letters.

"""

cities_list = ['manchester','karachi','islamabad','taxila','texas']
print([city.upper() for city in cities_list])

"""Task: 2
  1. Write a Python function that accepts a list of numbers and returns the maximum number.
"""

def max_finder(nums):
  nums_list = np.array(nums)
  return nums_list.max()

nums = [1,45435,567,87967,3]
print(max_finder(nums))

"""  2. Write another function that checks if a number is prime.

"""

def is_prime(number):
  #checking for tables in the range starting from
  #table of 2 till the table of square root of number
  for i in range(2,int(np.sqrt(number))+1):
    if number % i == 0:
      return False
  return True

num=34
print(is_prime(num))
 #The loop for number 2 is never executed because the loop for range
 #starting from 2 and ending at 2 will never be executed so without
 #visiting if block it returns directly True.

"""  3. Create a loop that calls the prime-checking function on numbers from 1 to 20 and prints whether each number is prime or not.

"""

#print([is_prime(num) for num in range(1,20+1)])
for num in range(1,20+1):
  print(f'{num}:{is_prime(num)}')

"""Task: 3
  1. Create a NumPy array of 10 random integers between 1 and 50.
 2. Calculate the sum, mean, and standard deviation of the array.

"""

import random

#only a list can be converted to array by numpy
array = np.array([random.randint(1,50) for num in range(10)])
print(f'Sum:{round(array.sum(),2)}')
print(f'Mean:{round(array.mean(),2)}')
print(f'Std Deviation:{round(array.std(),2)}')

#sorting in descending order thats why reverse is passed as True
#itherwise the default is reverse = False for ascending order
print(f'Sorted array:{sorted(array,reverse=True)}')

"""Task: 4
  1. Create a Pandas DataFrame with three columns: Name, Age, and Score.
  2. Add data for at least five students.

"""

student_dict = {}
Name = ['Azhar','Akram','Mehtab','Ghufran','Fehmeen']
Age = [21,32,23,45,22]
Score = [33,67,98,54,66]

for i in range(len(Name)):
  if 'Name' not in student_dict and 'Age' not in student_dict and 'Score' not in student_dict:
    student_dict['Name'] = [Name[i]]
    student_dict['Age'] = [Age[i]]
    student_dict['Score'] = [Score[i]]
  else:
    student_dict['Name'].append(Name[i])
    student_dict['Age'].append(Age[i])
    student_dict['Score'].append(Score[i])


df = pd.DataFrame(student_dict)
df

"""3. Calculate the average score and filter out students who scored below 70."""

#calculating average score
print(df['Score'].mean())

mask = df['Score']<70
df['Name'][mask]

"""4. Print the names of students who passed (scored 70 or above).

"""

mask = df['Score'] >= 70
df['Name'][mask]

"""
Task 5
  1. Create a new Colab notebook.
  2. Write Python code to generate the first 10 Fibonacci numbers."""

def generate(size):
  fib_list = [0,1]

  for i in range(2,size):
    fib_list.append(fib_list[i-2]+fib_list[i-1])
  return fib_list

print(generate(10))