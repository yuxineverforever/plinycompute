/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef SHARED_EMPLOYEE_H
#define SHARED_EMPLOYEE_H

#include "Object.h"
#include "PDBVector.h"
#include "PDBString.h"
#include "Handle.h"
#include <vector>

class SharedEmployee : public pdb::Object {

    int age;
    double salary;

public:
    pdb::Handle<pdb::String> name;

    ENABLE_DEEP_COPY

    ~SharedEmployee() = default;
    SharedEmployee() = default;

    void print() {
        std::cout << "name is: " << *name << " age is: " << age;
    }

    std::string toSchemaString(std::string format) {
        if (format == "csv") {
            return "name,age,salary\n";
        } else {
            return "";
        }
    }

    std::string toValueString(std::string format) {
        if (format == "csv") {
            char buffer[65535];
            sprintf(buffer, "%s,%d,%f\n", name->c_str(), age, salary);
            return buffer;
        } else {
            return "";
        }
    }


    std::vector<std::string> getSupportedFormats() {
        std::vector<std::string> ret;
        ret.push_back("csv");
        return ret;
    }

    pdb::Handle<pdb::String>& getName() {
        return name;
    }

    bool isFrank() {
        return (*name == "Frank");
    }

    SharedEmployee(std::string nameIn, int ageIn) {
        name = pdb::makeObject<pdb::String>(nameIn);
        age = ageIn;
        salary = 4000.0;
    }

    SharedEmployee(std::string nameIn, int ageIn, double salaryIn) {
        name = pdb::makeObject<pdb::String>(nameIn);
        age = ageIn;
        salary = salaryIn;
    }


    double getSalary() {
        return salary;
    }
};


#endif
