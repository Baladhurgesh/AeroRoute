INTEGER_FIELDS = {
    "origin_airport_id": "OriginAirportID", "origin_airport_seq_id": "OriginAirportSeqID",
    "destination_airport_id": "DestAirportID", "destination_airport_seq_id": "DestAirportSeqID",
    "scheduled_elapsed_minutes": "CRSElapsedTime", "departure_delay_minutes": "DepDelay",
    "arrival_delay_minutes": "ArrDelay", "actual_elapsed_minutes": "ActualElapsedTime",
    "diversion_elapsed_minutes": "DivActualElapsedTime", "diversion_arrival_delay_minutes": "DivArrDelay",
    "diversion_landings": "DivAirportLandings", "taxi_in_minutes": "TaxiIn", "taxi_out_minutes": "TaxiOut",
    "air_time_minutes": "AirTime", "marketing_carrier_dot_id": "DOT_ID_Marketing_Airline",
    "operating_carrier_dot_id": "DOT_ID_Operating_Airline",
}
STRING_FIELDS = {
    "origin": "Origin", "destination": "Dest", "origin_state_name": "OriginStateName",
    "destination_state_name": "DestStateName", "marketing_carrier": "Marketing_Airline_Network",
    "operating_carrier": "Operating_Airline", "marketing_flight_number": "Flight_Number_Marketing_Airline",
    "operating_flight_number": "Flight_Number_Operating_Airline", "duplicate_flag": "Duplicate",
    "originally_scheduled_carrier": "Originally_Scheduled_Code_Share_Airline",
    "originally_scheduled_flight_number": "Flight_Num_Originally_Scheduled_Code_Share_Airline",
    "codeshare_partner": "Operated_or_Branded_Code_Share_Partners", "cancellation_code": "CancellationCode",
    "scheduled_departure_clock": "CRSDepTime", "scheduled_arrival_clock": "CRSArrTime",
    "actual_departure_clock": "DepTime", "actual_arrival_clock": "ArrTime",
}
BOOLEAN_FIELDS = {"cancelled": "Cancelled", "diverted": "Diverted", "diversion_reached_destination": "DivReachedDest"}
TIME_FIELDS = ("scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival")
