from setuptools import setup, find_packages

setup(
    name="acllite",
    version="1.0",
    packages=find_packages(),
    # 如果 acllite 就是顶层包（没有子包），也可以写成：
    # py_modules=["acllite_model", "acllite_logger", "constants", ...],
    # 但 find_packages() 在有 __init__.py 时也能工作
)
