/*
* Software Name : SYRROCA
* Version: 1.0
* SPDX-FileCopyrightText: Copyright (c) 2021 Orange
* SPDX-License-Identifier: BSD-3-Clause
*
* This software is distributed under the BSD 3-Clause "New" or "Revised" License,
* the text of which is available at https://spdx.org/licenses/BSD-3-Clause.html
* or see the "license.txt" file for more details.
*
* Author: Alessio Diamanti
*/


class lstmObjts:
    def __init__(self, X_train_x, oderedColumn_x, scaler_x, feature_size_x):
        self.X_train_x = X_train_x
        self.oderedColumn_x = oderedColumn_x
        self.scaler_x = scaler_x
        self.feature_size_x = feature_size_x

class predictRes:
    def __init__(self, scored, errorDfSquare, predicted):
        self.scored = scored
        self.errorDfSquare = errorDfSquare
        self.predicted = predicted

class ErrorClass:
    def __init__(self, pattern = [], timeIndex = [], patternString = [], freq = 0):
        self.pattern = pattern
        self.patternString = patternString
        self.timeIndex = timeIndex
        self.containers = []
        self.freq = freq

    def set_containers(self, containers):
        self.containers.append(containers)

    def set_freq(self, freq):
        self.freq = freq

    def set_patternString(self, pttString):
        self.patternString = pttString

    def set_timeIndex(self, timeIdx):
        self.timeIndex = timeIdx

    def copy(self):
        return ErrorClass(self.pattern, self.timeIndex, self.patternString, self.freq)
