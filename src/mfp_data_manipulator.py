import pandas as pd
import numpy as np
import os, sys, re, warnings

warnings.filterwarnings("ignore")

print("Data Manipulator")
print("=" * 70)
print("Program to slice, sort, generate, append, delete, and modify data in a CSV file.")
print("=" * 70)

class MFPDataManipulator:
    def __init__(self, datafile=None):
        if datafile:
            self.datafile = os.path.abspath(datafile)  # Get absolute path
            self.df = pd.read_csv(self.datafile).replace(np.nan, '', regex=True)
            print ("Datafile loaded successfully.")
            print(self.df)
        else:
            self.datafile = None
            self.df = pd.DataFrame()  # Initialize an empty DataFrame

    def slice(self, start, end):
        start = int(start)
        end = int(end)
        self.df = self.df.iloc[start:end]  # Update self.df
        return self.df

    def sort(self, col, order):
        if order in ["ascending", "asc"]:
            self.df = self.df.sort_values(by=[col], ascending=True)  # Update self.df
        elif order in ["descending", "desc"]:
            self.df = self.df.sort_values(by=[col], ascending=False)  # Update self.df
        else:
            raise ValueError("Invalid order. Use 'asc' or 'desc'.")
        return self.df

    def generate(self, xr, expr):
        xr = xr.split(":")
        x = np.linspace(int(xr[0]), int(xr[1]), int(xr[1]) - int(xr[0]) + 1)
        y = eval(expr)
        new_df = pd.DataFrame({"x": x, "y": y})
        self.df = new_df  # Replace any previous data frame with the new one
        return self.df

    def append(self, datafile):
        df2 = pd.read_csv(datafile)
        df2 = df2.replace(np.nan, '', regex=True)
        self.df = pd.concat([self.df, df2])  # Update self.df
        return self.df

    def delete(self, targets):
        # Check if the input targets are columns (strings) or rows (integers)
        if all(target.strip().isnumeric() for target in targets.split(',')):
            # If all targets are numeric, treat them as row indices
            row_indices = [int(target.strip()) for target in targets.split(',')]
            self.df = self.df.drop(index=row_indices)  # Drop specified rows
        else:
            # Otherwise, treat them as column names
            col_names = [target.strip() for target in targets.split(',')]
            self.df = self.df.drop(columns=col_names)  # Drop specified columns
        return self.df

    def modify(self, col, old_val, new_val):
        self.df[col] = self.df[col].replace(old_val, new_val)  # Update self.df
        return self.df

    def save(self, datafile):
        directory = os.path.dirname(self.datafile)
        save_path = os.path.join(directory, datafile)
        self.df.to_csv(save_path, index=False)  # Save the updated DataFrame
        print(f"Data saved to {save_path}")

if __name__ == "__main__":
    datafile = input("Enter the CSV file name: ")
    print ("Datafile loaded successfully.")
    data_manipulator = MFPDataManipulator(datafile)

    while True:
        datafile = input("Enter the CSV file name (or press Enter to skip): ")
        if datafile.strip():
            try:
                data_manipulator = MFPDataManipulator(datafile)
                print("Datafile loaded successfully.")
            except FileNotFoundError:
                print(f"File '{datafile}' not found. Please try again.")
                continue
        else:
            data_manipulator = MFPDataManipulator()  # Create an instance with no file
            print("No datafile loaded. You can generate data from scratch.")
        
        while True:
            action = input("Enter an action (slice, sort, generate/gen, append, delete/del, modify, save, or exit/q): ").lower()
            
            if action == "generate" or action == "gen":
                xr = input("Enter the range for x (e.g., 0:10): ")
                expr = input("Enter the expression for y (e.g., 5*x**2/np.exp(x)): ")
                print(data_manipulator.generate(xr, expr))
            
            elif action == "slice":
                if data_manipulator.df.empty:
                    print("No data available to slice.")
                    continue
                start = input("Enter the start index: ")
                end = input("Enter the end index: ")
                print(data_manipulator.slice(start, end))
            
            elif action == "sort":
                if data_manipulator.df.empty:
                    print("No data available to sort.")
                    continue
                order = input("Enter the order (asc/desc): ")
                col = input("Enter the column name to sort by: ")
                print(data_manipulator.sort(col, order))
            
            elif action == "append":
                file_to_append = input("Enter the file name to append: ")
                print(data_manipulator.append(file_to_append))
            
            elif action == "delete" or action == "del":
                if data_manipulator.df.empty:
                    print("No data available to delete.")
                    continue
                targets = input("Enter the column names or row indices to delete (comma-separated): ")
                print(data_manipulator.delete(targets))
            
            elif action == "modify":
                if data_manipulator.df.empty:
                    print("No data available to modify.")
                    continue
                col = input("Enter the column name to modify: ")
                old_val = input("Enter the old value to replace: ")
                new_val = input("Enter the new value: ")
                print(data_manipulator.modify(col, old_val, new_val))
            
            elif action == "save":
                save_file = input("Enter the file name to save the data: ")
                data_manipulator.save(save_file)
                print(f"Data saved to {save_file}")
            
            elif action == "exit" or action == "q":
                print("Exiting the program.")
                exit(0)
            
            else:
                print("Invalid action. Please try again.")