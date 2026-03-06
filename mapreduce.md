```python
class Mapper:
    method Map(line_offset, line_content):
        // 1. Parse the input line into distinct fields
        student_id, course_name, grade, department = split(line_content, ",")
        
        // 2. Filter records to only include "Big Data Management"
        if course_name == "Big Data Management":
            
            // 3. Emit the department as the key. 
            // Emit a tuple containing the (grade, count) as the value.
            // We use 1 as the count to keep track of the number of students.
            emit(department, (grade, 1))


class Reducer:
    method Reduce(department, values_list):
        total_grade_sum = 0
        total_student_count = 0
        
        // 1. Iterate through the list of (grade, count) tuples for the department
        for each (grade, count) in values_list:
            total_grade_sum = total_grade_sum + grade
            total_student_count = total_student_count + count
            
        // 2. Compute the average grade
        average_grade = total_grade_sum / total_student_count
        
        // 3. Emit the final department and its calculated average
        emit(department, average_grade)
```

- Explanation of the Pseudocode
The Mapper processes the dataset line by line. It extracts the relevant attributes and applies a filter to ensure we are only analyzing the course "Big Data Management." If the record matches, it outputs a key-value pair where the key is the department and the value is a composite of the student's grade and a tally count of 1. Passing a count of 1 is crucial because a Reducer needs both the sum of the grades and the total number of grades to accurately calculate an average.

The Shuffle & Sort Phase (handled automatically by the MapReduce framework behind the scenes) gathers all the emitted values that share the same key (the same department) and passes them to the Reducer as a list.

The Reducer receives a specific department and a list of all the (grade, 1) tuples associated with it. It loops through this list, accumulating a running total of both the grades and the counts. Finally, it divides the total sum by the count to find the average and outputs the final result.

- Example Run Demonstration
Let's assume the following 4 lines of input data. This satisfies the requirement of having >2 instances, at least 2 instances of the same department, and I have included an irrelevant course to demonstrate the filtering step.

Input Dataset:

Line 1: 1, Big Data Management, 80, Computer Science

Line 2: 2, Big Data Management, 90, Computer Science

Line 3: 3, Introduction to Programming, 100, Computer Science

Line 4: 4, Big Data Management, 85, Data Analytics

Step 1: Map Phase
The Mapper reads each line independently:

Line 1: Matches course. Emits -> ("Computer Science", (80, 1))

Line 2: Matches course. Emits -> ("Computer Science", (90, 1))

Line 3: Fails the course_name == "Big Data Management" check. Filtered out.

Line 4: Matches course. Emits -> ("Data Analytics", (85, 1))

Step 2: Shuffle and Sort Phase
The framework groups the intermediate key-value pairs by their key (department):

"Computer Science" -> [(80, 1), (90, 1)]

"Data Analytics" -> [(85, 1)]

Step 3: Reduce Phase
The Reducer processes each grouped key-value list:

Reducer Instance A (Computer Science):

Iterates through [(80, 1), (90, 1)]

total_grade_sum = 80 + 90 = 170

total_student_count = 1 + 1 = 2

average_grade = 170 / 2 = 85

Emits final output -> ("Computer Science", 85)

Reducer Instance B (Data Analytics):

Iterates through [(85, 1)]

total_grade_sum = 85

total_student_count = 1

average_grade = 85 / 1 = 85

Emits final output -> ("Data Analytics", 85)