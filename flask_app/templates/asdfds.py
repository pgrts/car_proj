def index():
    matches = None  # Default to None if no search is performed
    year, make, model = None, None, None  # Defaults for conditional rendering

    if request.method == 'POST':
        action = request.form.get('action')  # Get the button action

        if action == 'search_by_vin':
            vin = str(request.form.get('vin')).strip()

            if vin:
                
                vehicle_row = create_assumption(df_vehicles[df_vehicles['vin'] == vin])
                if not vehicle_row.empty:

                    features = vehicle_row.iloc[0]
                else:
                    df = vin_decode(vin)
 
                    if df.empty:
                        raise ValueError("Error: No vehicle information found for the provided VIN.")

                    df = df.replace('', 'nan')  # Replace empty strings with 'nan'

                    features = create_assumption(df)

                similar_vehicles = find_similar_vehicles(features, model_prep(df_vehicles))
                plot_url, results = plot_comparison(features, similar_vehicles)
                results_html = results.to_html(index=False, classes='data', border=0)

                return render_template('result.html', plot_url=plot_url, results=results_html, similar_vehicles=similar_vehicles[[x.lower() for x in ['Make', 'Model', 'ModelYear', 'Series', 'Trim', 'DriveType']]])

            else:
                error_message = "Please enter a VIN."
                return render_template('index.html', error_message=error_message)

        elif action == 'search_by_make_model_year':
            make_model_year = str(request.form.get('make_model_year')).strip()
            if make_model_year:
                # Parse make, model, and year from input
                try:
                    parts = make_model_year.split()
                    year = int(parts[-1])  # Assuming the year is always the last part
                    make = parts[0]
                    model = " ".join(parts[1:-1])
                except (ValueError, IndexError):
                    error_message = "Please enter a valid Make, Model, and Year."
                    return render_template('index.html', error_message=error_message)

                # Perform search in your df_vehicles DataFrame
                matches = df_vehicles[
                    (df_vehicles['make'].str.lower() == make.lower()) &
                    (df_vehicles['model'].str.lower() == model.lower()) &
                    (df_vehicles['modelyear'] == year)
                ]

                if matches.empty:
                    error_message = f"No vehicles found for {year} {make} {model}."
                    return render_template('index.html', error_message=error_message)

                # Matches are found; pass them to the template
                return render_template('index.html', matches=matches, year=year, make=make, model=model)

    return render_template('index.html', matches=matches)



def search_make_model():
    make_model_year = request.args.get('make_model_year')  # Use request.args for GET parameters
    parts = make_model_year.split(' ')
    year, make, model = None, None, None

    for part in parts:
        if part.isnumeric():
            year = float(part)
        elif not make:
            make = part
        else:
            model = part

    make_lower = make.lower() if make else None

    matches = df_vehicles[
        (df_vehicles['make'].str.lower() == make_lower) &
        (df_vehicles['model'].str.contains(model, case=False)) &
        (df_vehicles['modelyear'] == year)
    ]
    
    if matches.empty:
        return render_template('error.html', error_message="No vehicles found matching the provided Make, Model, and Year.")
    # Get unique Series, Trim, and DriveType options along with their index
    unique_options = matches[[x.lower() for x in ['series', 'trim', 'drivetype', 'displacementcc', 'fueltypeprimary']]].drop_duplicates()
    return render_template('select_vehicle.html', matches=unique_options, make=make_lower, model=model, year=year)