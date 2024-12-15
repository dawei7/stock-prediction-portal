import React, {useEffect} from 'react'
import axios from 'axios'
import axiosInstance from '../../axiosinstance'

const Dashboard = () => {
  useEffect(() => {
    const fetchProtectedData = async () => {
      try {
        const response = await axiosInstance.get("/protected-view/")
        console.log("Success", response.data)
      } catch (error) {
        console.log(error)
      }
    }
    fetchProtectedData()
  }, [])
  return (
    <>
        <div className="container">
            <div className='p-5 text-center bg-light-dark rounded'>
                <h1 className='text-light'>Stock Prediction Portal</h1>
                <p className='text-light lead'>Welcome to the Stock Prediction Portal</p>
            </div>
        </div>
    </>
  )
}

export default Dashboard