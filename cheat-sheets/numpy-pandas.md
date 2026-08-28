# NumPy and Pandas Cheat Sheet

## NumPy
| Task | Code |
|------|------|
| Create array | `np.array([1,2,3])` |
| Zeros / ones | `np.zeros((2,3))`, `np.ones(3)` |
| Range | `np.arange(0,10,2)` |
| Reshape | `a.reshape(3,2)` |
| Mean / std | `a.mean()`, `a.std()` |
| Matrix multiply | `a @ b` |

## Pandas
| Task | Code |
|------|------|
| Read CSV | `pd.read_csv('f.csv')` |
| First rows | `df.head()` |
| Info | `df.info()`, `df.describe()` |
| Missing | `df.isna().sum()`, `df.fillna(0)` |
| Filter | `df[df['age'] > 30]` |
| Group | `df.groupby('c')['v'].mean()` |
| Merge | `pd.merge(a, b, on='id')` |
