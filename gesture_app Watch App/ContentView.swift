

import SwiftUI
import CoreML
import CoreMotion
//main関数の呼び出す回数と，センサのサンプリング周波数は同じであるが，処理の重さなどの外的要因により，インクリメントがずれる
struct ContentView: View {
    //IMU sensor
    @ObservedObject var sensor = MotionSensor()
    
    func print_Result() -> Text{
        if sensor.gesture_pose == "neutral"{
            return Text("Gesture:neutral")
        }else if sensor.gesture_pose == "clunch"{
            return Text("Gesture:crunch")
        }
        return Text("Gesture:")
    }
    
    
    
    var body: some View {
        VStack {

            //Button (ON:ジェスチャ認識開始 OFF:ジェスチャ認識停止)
//            print_Result()
            Button(action:{
//                if sensor.isStarted {
//                    sensor.stop()
//                }else{
//
//                    sensor.start()
//
//                }

                sensor.start()
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) {
                    sensor.stop()
                }
                

            }){
                sensor.isStarted ? Text("Predict..."):Text("START")
            }
            
            
        }
        
        
    }
    
    
}

class GesturesClassifier{
    //Create arrays for aggregating inputs
    struct ModelConstants{
        static let predictionWindowSize = 50
        static let sensorsUpdateInterval = 1.0/100.0
        static let stateInLength = 400
    }
    
    //acceleration
    var accelDataX = try! MLMultiArray(shape:[ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let accelDataY = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let accelDataZ = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    //gyro sensor
    let gyroDataX = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let gyroDataY = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    let gyroDataZ = try! MLMultiArray(shape: [ModelConstants.predictionWindowSize] as [NSNumber], dataType: MLMultiArrayDataType.double)
    
    var stateOutput = try! MLMultiArray(shape: [ModelConstants.stateInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    
    var currentIndexInPredictionWindow: Int = 0
    
    var t_stamp = 0.0
    
    //gesture pose
    @Published var gesture_pose: String = ""
    
    var model:Gestures_3Classifier_nue2
    
    init(){
        model = Gestures_3Classifier_nue2()
        
    }
    
    func init_array(){
        currentIndexInPredictionWindow = 0
//        for i in 0..<ModelConstants.predictionWindowSize{
//            accelDataX[i] = 0.0 as NSNumber
//            accelDataY[i] = 0.0 as NSNumber
//            accelDataZ[i] = 0.0 as NSNumber
//            gyroDataX[i] = 0.0 as NSNumber
//            gyroDataY[i] = 0.0 as NSNumber
//            gyroDataZ[i] = 0.0 as NSNumber
//        }
    }
    
    func process(device: CMDeviceMotion){
        
        if currentIndexInPredictionWindow == ModelConstants.predictionWindowSize{
            return
        }
        
        accelDataX[[currentIndexInPredictionWindow] as [NSNumber]] = device.userAcceleration.x as NSNumber
        accelDataY[[currentIndexInPredictionWindow] as [NSNumber]] = device.userAcceleration.y as NSNumber
        accelDataZ[[currentIndexInPredictionWindow] as [NSNumber]] = device.userAcceleration.z as NSNumber
        gyroDataX[[currentIndexInPredictionWindow] as [NSNumber]] = device.rotationRate.x as NSNumber
        gyroDataY[[currentIndexInPredictionWindow] as [NSNumber]] = device.rotationRate.y as NSNumber
        gyroDataZ[[currentIndexInPredictionWindow] as [NSNumber]] = device.rotationRate.z as NSNumber
        
        
        currentIndexInPredictionWindow += 1
        
        print(currentIndexInPredictionWindow)
//        print(accelDataX[0])
//        print(self.detect_peak(data: accelDataX, num_train: 20, num_guard: 10, rate_fate: 1e-3))
        if currentIndexInPredictionWindow == ModelConstants.predictionWindowSize{
//        if currentIndexInPredictionWindow == ModelConstants.predictionWindowSize && self.detect_peak(data: accelDataX, num_train: 10, num_guard: 2, rate_fate: 0.001) != -1 {
            DispatchQueue.global().async {
                self.perfomModelPrediction()
//                print(self.perfomModelPrediction())
                DispatchQueue.main.async {
                    self.currentIndexInPredictionWindow = 0
                }
            }
        }
        
        
        
    }
    
    func addSampleToDataArray(Sample: MotionSensor)-> Text{
        
        currentIndexInPredictionWindow = Sample.cnt
        accelDataX[[currentIndexInPredictionWindow] as [NSNumber]] = Sample.acceX as NSNumber
        accelDataY[[currentIndexInPredictionWindow] as [NSNumber]] = Sample.acceY as NSNumber
        accelDataZ[[currentIndexInPredictionWindow] as [NSNumber]] = Sample.acceZ as NSNumber
        gyroDataX[[currentIndexInPredictionWindow] as [NSNumber]] = Sample.rotX as NSNumber
        gyroDataY[[currentIndexInPredictionWindow] as [NSNumber]] = Sample.rotY as NSNumber
        gyroDataZ[[currentIndexInPredictionWindow] as [NSNumber]] = Sample.rotZ as NSNumber
       
//        print(accelDataX)
//        print(Sample.cnt)
        print(accelDataX[0])
        if currentIndexInPredictionWindow == ModelConstants.predictionWindowSize{
            
            let predictedActivity = perfomModelPrediction()
                //初期化


                //
//                print(predictedActivity)
                
                

        }
//        print("time:\(Sample.timestamp)-\(t_stamp)\n")
//        print("currentIndexInPredictionWindow:\(currentIndexInPredictionWindow)\n")
//
//        if(currentIndexInPredictionWindow % ModelConstants.predictionWindowSize == 0){
//            if let predictedActivity = perfomModelPrediction(model:self.model){
//                //初期化
//
//                Sample.cnt = 0
//
//                //
//                if predictedActivity == "pinch_3"{
//                    gesture_pose = "pinch"
//                }else if predictedActivity == "neutral_3"{
//                    gesture_pose = "neutral"
//                }
//
//                print(Sample.cnt)
//
//
//            }
//        }
        
        return Text(gesture_pose)
    }
    
    private func argmax(data:MLMultiArray,first:Int,last:Int) -> Int{
        var max:Double = 0.0
        var maxi = 0
        for i in first..<last{
            if max < Double(data[i]){
                max = Double(data[i])
                maxi = i
            }
        }
        return maxi
        
    }
    
    private func npsum(data:MLMultiArray,first:Int,last:Int) -> Double{
        var sum:Double = 0.0
        
        for i in first..<last{
            sum+=Double(data[i])
        }
        
        return sum
    }
    
    private func detect_peak(data: MLMultiArray,num_train:Int,num_guard:Int,rate_fate:Double) -> Int{
        let num_cells = ModelConstants.predictionWindowSize
        let num_train_half = round(Double(num_train)/2)
        let num_guard_half = round(Double(num_guard)/2)
        let num_side = Int(num_train_half + num_guard_half)
        
        let alpha = Double(num_train) * (pow(rate_fate,Double(-1/Double(num_train)))-1)
        
        
        
        for i in num_side..<(num_cells - num_side) {
              
            if i != self.argmax(data:data, first:i-num_side,last:i+num_side){
                continue
            }
            
            let sum1 = npsum(data: data, first: i-num_side, last: i+num_side+1)
            let sum2 = npsum(data: data, first: i-Int(num_guard_half), last: i+Int(num_guard_half)+1)
            
            let p_noise = (sum1 - sum2) / Double(num_train)
            
            let threshold = alpha * p_noise
            
            if Double(data[i]) > threshold{
                return i
            }
        }
        return -1
    }
    
    func perfomModelPrediction() -> [(String, Double)]{
        
        let modelPrediction = try! model.prediction(accex: accelDataX, accey: accelDataY, accez: accelDataZ, gyrox: gyroDataX, gyroy: gyroDataY, gyroz: gyroDataZ, stateIn: stateOutput)
        
        stateOutput = modelPrediction.stateOut
//        print(modelPrediction.labelProbability.values)
        let sorted = modelPrediction.labelProbability.sorted{
            return $0.value > $1.value
        }
        print(sorted[0].0)
        print(sorted[0].1)
//        print(sorted[1].0)
//        print(sorted[1].1)
//        print(sorted[2].0)
//        print(sorted[2].1)
        self.gesture_pose = sorted[0].0
        return sorted
    }
    
    
}


class MotionSensor: NSObject, ObservableObject{
    let motionManager = CMMotionManager()
    // Load the CoreML model
    let model = GesturesClassifier()
    
    @Published var isStarted = false
    
    @Published var acceX = 0.0
    @Published var acceY = 0.0
    @Published var acceZ = 0.0
    
    @Published var rotX = 0.0
    @Published var rotY = 0.0
    @Published var rotZ = 0.0
    
    @Published var timestamp = 0.0
    @Published var gesture_pose = ""
    @Published var cnt: Int = 0
    func start(){
        self.model.init_array()
       
        if self.motionManager.isDeviceMotionAvailable{
            self.motionManager.deviceMotionUpdateInterval = 0.01
            self.motionManager.startDeviceMotionUpdates(to: OperationQueue.main) { (motion, error) in
                if let motion = motion{
//                    print(motion)
                    self.model.process(device: motion)
                }
                //                self.updateMotionData(deviceMotion: motion!)})
            }
        }
        
        self.isStarted = true
        
        
        
    }
    
    func stop(){
        isStarted = false
        motionManager.stopDeviceMotionUpdates()
    }
    
    func print_result()->Text{
        return Text(model.gesture_pose)
    }
    
//    private func updateMotionData(deviceMotion:CMDeviceMotion){
//
//        cnt += 1
//        acceX = deviceMotion.userAcceleration.x
//        acceY = deviceMotion.userAcceleration.y
//        acceZ = deviceMotion.userAcceleration.z
//        rotX = deviceMotion.rotationRate.x
//        rotY = deviceMotion.rotationRate.y
//        rotZ = deviceMotion.rotationRate.z
//        model.addSampleToDataArray(Sample: self)
//
//
//
//    }
    
}
