import pandas as pd

df = pd.read_csv("/mnt/storage/jiwei/DGNNCL/Data/Movie.csv")
print(df.head())

user_sequences = df.groupby("user_id")["item_id"].apply(list)

total_users = len(user_sequences)
users_with_duplicates = 0

for items in user_sequences:
    if len(items) != len(set(items)):
        users_with_duplicates += 1

ratio = users_with_duplicates / total_users

print(f"总用户数: {total_users}")
print(f"有重复交互 item 的用户数: {users_with_duplicates}")
print(f"比例: {ratio:.4f}  （约 {ratio*100:.2f}% ）")
