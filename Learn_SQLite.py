import sqlite3

#Create an employee class
class Employee:
    """A sample Employee class"""
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
    
    @property
    def email(self):
        return '{}.{}'.format(self.first, self.last)
    
    def __repr__(self):
        return "Employee('{}','{}',{})".format(self.first, self.last, self.pay)

'''Connect to employee.db file'''
#If we want a permanent database
#conn = sqlite3.connect('employee.db')

#If we want to store the database in memory - good for testing
#if we want to run a fresh database on every run
conn = sqlite3.connect(':memory:')

'''Create a cursor - allows us to execute sql commands'''
c = conn.cursor()

'''Create employee table'''
c.execute("""CREATE TABLE employees (
            first  text,
            last text,
           pay integer
            )""")


'''Add data to database'''
#c.execute("INSERT INTO employees VALUES ('Bob', 'Marley', 50000)")

#Pull data that we have just entered
#c.execute("SELECT * FROM employees WHERE last = 'Marley'")

#To return one row
#c.fetchone()

#To return a list - it will return an empty list if there are no more rows
#c.fetchmany(5)

#To return all as a list - it will return an empty list if there are no more rows
#c.fetchall()

'''Add a new name, then pull the name'''
#c.execute("INSERT INTO employees VALUES ('Mary', 'Marley', 70000)")
#conn.commit()

#c.execute("SELECT * FROM employees WHERE last = 'Marley'")
#c.fetchall()


'''Inserting new employees into the database'''
#emp_1 = Employee('John', 'Doe', 80000)
#emp_2 = Employee('Jane', 'Doe', 90000)

#print(emp_1.first)
#print(emp_1.last)
#print(emp_1.pay)

#Two ways to insert the employee names into the database
#First way
#c.execute("INSERT INTO employees VALUES (?, ?, ?)", (emp_1.first, emp_1.last, emp_1.pay))

#Second way
#c.execute("INSERT INTO employees VALUES (:first, :last, :pay)", {'first': emp_2.first, 'last': emp_2.last, 'pay': emp_2.pay})

#Pull a name
#When using the first approach
#c.execute("SELECT * FROM employees WHERE last=?", ('Marley',))

#When using the second approach
#c.execute("SELECT * FROM employees WHERE last=:last", {'last': 'Doe'})

#print(c.fetchall())

'''Create a set of functions to automate inserting, updating and removing employees'''
def insert_emp(emp):
    with conn: #By using conn (context manager), we don't need a commit statement afterwards
        c.execute("INSERT INTO employees VALUES (:first, :last, :pay)", {'first': emp.first, 'last': emp.last, 'pay': emp.pay})
    
def get_emps_by_name(lastname):
    c.execute("SELECT * FROM employees WHERE last=:last", {'last': lastname})
    return c.fetchall()

def update_pay(emp, pay):
    with conn:
        c.execute("""UPDATE employees SET pay = :pay
            WHERE first = :first AND last = :last""",
            {'first': emp.first, 'last': emp.last, 'pay': pay})

def remove_emp(emp):
    with conn:
        c.execute("DELETE from employees WHERE first = :first AND last = :last",
                  {'first': emp.first, 'last': emp.last})

'''Insert a couple of employees'''
emp_1 = Employee('John', 'Doe', 80000)
emp_2 = Employee('Jane', 'Doe', 90000)


insert_emp(emp_1)
insert_emp(emp_2)

emps = get_emps_by_name('Doe')
print(emps)

'''Update employee 2's pay and delete employee 1'''
update_pay(emp_2, 95000)
remove_emp(emp_1)

emps = get_emps_by_name('Doe')
print(emps)

conn.close()
