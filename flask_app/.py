
    # Filter df_vehicles by matching the key features
    matched_row = df_vehicles.loc[
        (df_vehicles["make"] == make) &
        (df_vehicles["model"] == model) &
        (df_vehicles['modelyear'] == float(modelyear)) &
        (df_vehicles["series"] == series) &
        (df_vehicles["trim"] == trim) &
        (df_vehicles["fueltypeprimary"] == fueltypeprimary) &
        (df_vehicles["displacementcc"] == float(displacementcc)) &
        (df_vehicles["drivetype"] == drivetype) &
        (df_vehicles["enginecylinders"] == enginecylinders)
    ]

    # Features to exclude when looking for other_feats
    exclude_feats = [
        "make", "model", "series", "trim", "fueltypeprimary",
        "displacementcc", "drivetype", "enginecylinders",
        "condition", "paint_color", "state", "state_income", "region", "odometer"
    ]

    # All categorical and numerical columns (replace `cats` and `nums` with actual lists)
    all_features = cats + nums

    # Determine other_feats by excluding specified columns
    other_feats = [feat for feat in all_features if feat not in exclude_feats]

    # Retrieve the corresponding values from the matched row for other_feats
    if not matched_row.empty:
        matched_values = matched_row[other_feats].iloc[0].to_dict()  # Use .iloc[0] to get the first match
    else:
        return jsonify({"error": "No matching vehicle found in database"}), 404

    # Combine with existing input features
   # full_features = {**features, **matched_values}

    # Call your prediction model (replace with your actual model prediction code)
    #predicted_price = cb72.predict(pd.DataFrame([full_features]))[0].round().astype(int)