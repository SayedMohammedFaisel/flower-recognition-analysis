from django.shortcuts import render

# Create your views here.

        
from django.shortcuts import render


def anime(request):
    if (request.method == "POST"):
        data = request.POST
        Area_Code = data.get('area_code')
        Locality_Code = data.get('locality_code')
        Region_Code = data.get('region_code')
        Height = data.get('height')
        Diameter = data.get('diameter')
        Species = data.get('species_input')

        if 'buttonpredict' in request.POST:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import confusion_matrix

            path = "C:\\Users\\sayed_xc0opgu\\OneDrive\\Desktop\\Train.csv"
            data = pd.read_csv(path)

            numerical_columns = ['Area_Code', 'Locality_Code', 'Region_Code', 'Height', 'Diameter']
            for column in numerical_columns:
                data[column] = data[column].fillna(data[column].median())

            data['Species'] = data['Species'].fillna(data['Species'].mode()[0])

            inputs = data.drop(['Class'], axis=1)
            output = data['Class']

            label_encoder = LabelEncoder()
            label_encoder.fit(inputs['Species'])  
            inputs['Species'] = label_encoder.transform(inputs['Species'])

            x_train, x_test, y_train, y_test = train_test_split(inputs, output, train_size=0.9)

            numerical_columns_to_scale = ['Area_Code', 'Locality_Code', 'Region_Code', 'Height', 'Diameter']
            sc = StandardScaler()
            x_train[numerical_columns_to_scale] = sc.fit_transform(x_train[numerical_columns_to_scale])

            model = GaussianNB()
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            cm = confusion_matrix(y_test, y_pred)

            new_data_point = [
                float(Area_Code) if Area_Code is not None else 0.0,
                float(Locality_Code) if Locality_Code is not None else 0.0,
                float(Region_Code) if Region_Code is not None else 0.0,
                float(Height) if Height is not None else 0.0,
                float(Diameter) if Diameter is not None else 0.0,
            ]

            try:
                new_data_point.append(label_encoder.transform([Species])[0])
            except ValueError:
                new_data_point.append(0)  

            if len(new_data_point) == x_train.shape[1]:
                numerical_input = new_data_point[:-1]  
                numerical_input = sc.transform([numerical_input])  
                new_data_point[:-1] = numerical_input[0]  
                predicted_Class = model.predict([new_data_point])
                result = predicted_Class[0]
                return render(request, 'anime.html', context={'result': result})
            else:
                result = "Error: Incorrect number of features in the user input."
                return render(request, 'flower.html', context={'result': result})
    return render(request, 'anime.html')










