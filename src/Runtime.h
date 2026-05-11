
#pragma once

#include <string>
#include <unordered_map>
#include <map>

using namespace std;

struct Timings{

	int historySize = 60;
	uint64_t counter = 0;

	map<string, vector<float>> entries;

	void add(string label, float milliseconds){

		entries[label].resize(historySize);

		int entryPos = counter % historySize;

		entries[label][entryPos] += milliseconds;
	}

	void newFrame(){

		counter++;
		int entryPos = counter % historySize;

		for(auto& [label, list] : entries){
			list[entryPos] = 0.0f;
		}
	}

	float getAverage(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float sum = 0.0f;
		for(float value : entries[label]){
			sum += value;
		}

		float avg = sum / historySize;

		return avg;
	}

	float getMean(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		vector<float> values = entries[label]; // makes a copy before sorting
		std::sort(values.begin(), values.end());

		return values[values.size() / 2];
	}

	float getMin(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float min = 1000000000.0f;
		for(float value : entries[label]){
			if(value == 0.0f) continue;
			min = std::min(min, value);
		}

		if(min == 1000000000.0f) return 0.0f;

		return min;
	}

	float getMax(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float max = 0.0f;
		for(float value : entries[label]){
			max = std::max(max, value);
		}

		return max;
	}

	float getMedianOfMaxOver60Frames(string label){
		if(entries.find(label) == entries.end()){
			return 0.0f;
		}

		float max = 0.0f;
		for(float value : entries[label]){
			max = std::max(max, value);
		}

		return max;
	}

};

struct StartStop{
	uint64_t t_start;
	uint64_t t_end;
};

struct Runtime{

	struct Timing{
		string label;
		float milliseconds;
	};
	inline static bool measureTimings;
	inline static Timings timings;
	inline static vector<StartStop> profileTimings;

	Runtime(){
		
	}

};