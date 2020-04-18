from datetime import date

def get_todays_date():
	today = date.today()
	return today.strftime("%m_%d_%Y")