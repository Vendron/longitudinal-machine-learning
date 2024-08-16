FROM python:3.9.19

WORKDIR /app

RUN pip install Scikit-longitudinal 
RUN pip uninstall scikit-learn -y && pip install scikit-lexicographical-trees

COPY . .

RUN ls

CMD ["python", "app.py"]