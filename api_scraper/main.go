package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/gocarina/gocsv"
)

type UsdaPayload struct {
	Data []FoodData `json:"data,omitempty"`
}

type FoodData struct {
	Description     string         `json:"description,omitempty" csv:"description"`
	DataType        string         `json:"dataType,omitempty" csv:"data_type"`
	PublicationDate string         `json:"publicationDate,omitempty" csv:"publication_date"`
	FoodNutrients   []NutrientData `json:"foodNutrients,omitempty" csv:"food_nutrients"`
	FdcId           int            `json:"fdcId,omitempty" csv:"fdc_id"`
}

type NutrientData struct {
	Number   string  `json:"number,omitempty" csv:"number"`
	Name     string  `json:"name,omitempty" csv:"name"`
	UnitName string  `json:"unitName,omitempty" csv:"unit_name"`
	Amount   float64 `json:"amount,omitempty" csv:"amount"`
}

type FoodTrainData struct {
	Name           string              `csv:"name"`
	Macronutrients []NutrientTrainData `csv:"macronutrients"`
}

type NutrientTrainData struct {
	Nutrient string  `csv:"nutrient"`
	Unit     string  `csv:"unit"`
	Amount   float64 `csv:"amount"`
}

// TODO: determine which food item to fetch
func FetchFoodData() ([]FoodData, error) {
	uri := fmt.Sprintf("https://api.nal.usda.gov/fdc/v1/foods/list?api_key=%s", "MrNLsnwrccxdxsBfffLPfaTQ29QfhpowdhebG9sJ")
	res, err := http.Get(uri)
	if err != nil {
		slog.Error("error fetching food data", "error_message", err.Error())
		return nil, err
	}
	defer res.Body.Close()

	body, err := io.ReadAll(res.Body)
	if err != nil {
		slog.Error("error reading response body payload", "error_message", err.Error())
		return nil, err
	}

	var data []FoodData
	err = json.Unmarshal(body, &data)
	if err != nil {
		slog.Error("error parsing json response", "error_message", err.Error())
		return nil, err
	}

	return data, nil
}

func ConstructTrainData(apiData []FoodData) ([]FoodTrainData, error) {
	wg := sync.WaitGroup{}
	foodChan := make(chan FoodTrainData, len(apiData))
	requiredMacronutrients := []string{"Protein", "Carbohydrate, by difference", "Total lipid (fat)", "Energy"}

	for _, entry := range apiData {
		wg.Add(1)
		go func(fd FoodData) {
			defer wg.Done()
			food := FoodTrainData{
				Name: fd.Description,
			}
			for _, n := range fd.FoodNutrients {
				if slices.Contains(requiredMacronutrients, n.Name) {
					nutrient := NutrientTrainData{
						Nutrient: n.Name,
						Amount:   n.Amount,
						Unit:     n.UnitName,
					}
					food.Macronutrients = append(food.Macronutrients, nutrient)
				}
			}
			foodChan <- food
		}(entry)
	}

	wg.Wait()
	close(foodChan)

	csvData := make([]FoodTrainData, 0)

	for food := range foodChan {
		csvData = append(csvData, food)
	}

	return csvData, nil
}

func main() {
	// construct data
	startTime := time.Now()
	file, err := os.Create("usda_data.csv")
	if err != nil {
		slog.Error("error creating csv file", "error_message", err.Error())
		return
	}
	defer file.Close()

	apiData, err := FetchFoodData()
	if err != nil {
		slog.Error("error fetching food data", "error_message", err.Error())
		return
	}

	csvData, err := ConstructTrainData(apiData)
	if err != nil {
		slog.Error("error constructing train data", "error_message", err.Error())
		return
	}

	if err := gocsv.MarshalFile(&csvData, file); err != nil {
		slog.Error("error parsing csv", "error_message", err.Error())
		return
	}
	slog.Info("usda food nutrient data scraping", "time taken to process", time.Since(startTime))
}
