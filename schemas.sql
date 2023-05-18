CREATE TABLE salaries(
	record_id SERIAL PRIMARY KEY NOT NULL,
	timestamp_ DATE NOT NULL,
	date_ DATE NOT NULL,
	company VARCHAR(100),
	level_ VARCHAR(100),
	title VARCHAR(100),
	total_yearly_compensation INT,
	location_ VARCHAR(100),
	latitude FLOAT,
	longitude FLOAT,
	years_of_experience FLOAT,
	years_at_company FLOAT,
	base_salary INT,
	stock_grant_value INT,
	bonus INT,
	gender VARCHAR(100),
	race VARCHAR(100),
	education VARCHAR(100)
);

CREATE TABLE layoffs(
	record_id SERIAL PRIMARY KEY NOT NULL,
	company VARCHAR(100),
	location_ VARCHAR(100),
	industry VARCHAR(100),
	total_laid_off INT,
	percentage_laid_off FLOAT,
	date_ DATE,
	stage VARCHAR(100),
	country VARCHAR(100)
);