import os
import pandas as pd
from openpyxl import load_workbook

input_folder = r"D:\csv\drowsiness\RGB"     
output_folder = r"D:\csv1\drowsiness\RGB" 
os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".csv"):
        csv_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
            if df.shape[1] == 1:
                df = pd.read_csv(csv_path, sep=",", encoding="utf-8-sig")
            excel_name = os.path.splitext(filename)[0] + ".xlsx"
            excel_path = os.path.join(output_folder, excel_name)

            df.to_excel(excel_path, index=False)
            wb = load_workbook(excel_path)
            ws = wb.active

            for col in ws.columns:
                max_length = 0
                col_letter = col[0].column_letter

                for cell in col:
                    if cell.value is not None:
                        max_length = max(max_length, len(str(cell.value)))

                ws.column_dimensions[col_letter].width = max_length + 2

            wb.save(excel_path)
            print(f"Completed: {filename} -> {excel_name}")

        except Exception as e:
            print(f"Error: {filename} -> {e}")
print("All done.")