// ------------------- Charts ------------------- 
// Bar Chart for layoff by industry
var barChartOptions = {
    series: [{
    name: "2020",
    data: [561.0, 896.0, 6063.0, 91.0, 1268.0, 685.0, 167.0, 8624.0, 3142.0, 6218.0, 782.0, 0.0, 1125.0, 377.0, 270.0, 1318.0, 0.0, 2958.0, 1150.0, 466.0, 199.0, 3699.0, 2382.0, 8002.0, 1304.0, 288.0, 324.0, 14656.0, 13983.0]
    }, {
    name: "2021",
    data: [0.0, 2434.0, 3600.0, 0.0, 90.0, 1943.0, 0.0, 0.0, 0.0, 2644.0, 80.0, 0.0, 0.0, 160.0, 0.0, 0.0, 0.0, 21.0, 83.0, 515.0, 22.0, 2900.0, 13.0, 1088.0, 0.0, 30.0, 0.0, 200.0, 0.0]
    }, {
    name: "2022",
    data: [100.0, 433.0, 19856.0, 8263.0, 1314.0, 8728.0, 635.0, 12999.0, 5546.0, 11288.0, 738.0, 3605.0, 15058.0, 4916.0, 350.0, 1473.0, 0.0, 3597.0, 1879.0, 6296.0, 861.0, 9932.0, 380.0, 20914.0, 2601.0, 3699.0, 1436.0, 15977.0, 1637.0]
    }, {
    name: "2023",
    data: [0.0, 100.0, 26821.0, 2431.0, 3227.0, 2813.0, 0.0, 8959.0, 210.0, 13812.0, 4299.0, 21603.0, 11491.0, 422.0, 216.0, 1235.0, 158.0, 3902.0, 2757.0, 31531.0, 151.0, 1795.0, 115.0, 24821.0, 9311.0, 2753.0, 1763.0, 6859.0, 1539.0]
    }],
    chart: {
        type: 'bar',
        height: 350,
        stacked: true,
        toolbar: {
            show: false
        },
        zoom: {
            enabled: true
        }
    },
    responsive: [{
        breakpoint: 480,
        options: {
          legend: {
            position: 'bottom',
            offsetX: -10,
            offsetY: 0
          }
        }
    }],
    dataLabels: {
        enabled: false,
    },
    plotOptions: {
        bar: {
            horizontal: false,
        }
    },
    xaxis: {
        categories: ["Aerospace", "Construction", "Consumer", "Crypto", "Data", "Education",
        "Energy", "Finance", "Fitness", "Food", "HR", "Hardware", "Healthcare",
        "Infrastructure", "Legal", "Logistics", "Manufacturing", "Marketing",
        "Media", "Other", "Product", "Real Estate", "Recruiting", "Retail",
        "Sales", "Security", "Support", "Transportation", "Travel"],
    },
    yaxis: {
        title: {
            text: "Number of People Laid Off"
        }
    },
    legend: {
        position: 'right',
        offsetY: 40
    },
    fill: {
        opacity: 1
    }
};


var barChart = new ApexCharts(document.querySelector("#industryChart"), barChartOptions);
barChart.render();

// Bar chart for top 20 highest pay companies
var barChartOptions2 = {
  series: [{
      name: "Salary",
      data: [500000.0, 493000.0, 481376.8656716418, 414333.3333333333, 411166.6666666667, 404562.5, 395915.25423728814, 390625.0, 387952.38095238095, 372000.0, 369913.04347826086, 366297.2972972973, 365300.5780346821, 365291.41104294476, 364800.0, 364481.7518248175, 364284.4827586207, 360931.0344827586, 354875.0, 354050.0]
  }],
  chart: {
      type: 'bar',
      height: 350,
      toolbar: {
          show: false
      },
  },
  colors: [
      "#f5b74f",
  ],
  plotOptions: {
      bar: {
          distributed: true,
          borderRadius: 1,
          horizontal: false,
          columnWidth: '40%',
      }
  },
  dataLabels: {
      enabled: false
  },
  legend: {
      show: false
  },
  xaxis: {
      categories: ['PDT Partners', 'CloudKitchens', 'Netflix', 'Snapchat', 'Snap',
      'Jane Street Capital', 'Roblox', 'Ripple', 'Brex', 'Chime', 'UiPath',
      'Slack', 'Stripe', 'Lyft', 'Upwork', 'Pinterest', 'Airbnb', 'Coupang',
      'Zapier', 'Snowflake'],
  },
  yaxis: {
      title: {
          text: "Average Annual Compensation"
      },
      labels: {
        formatter: function(value) {
          var val = Math.abs(value)
          if (val >= 1000) {
            val = (val / 1000).toFixed(0) + ' K'
          }
          return val
        }
      }
  }

};

var barChart2 = new ApexCharts(document.querySelector("#myChart2"), barChartOptions2);
barChart2.render();

// Area Chart
var areaChartOptions = {
    series: [{
        name: 'All Layoffs',
        data: [9628, 60141, 9690, 1539, 7728, 2695, 2108, 3292, 9909, 35257, 35159, 84186, 166647, 18489]
    }, {
        name: 'STEM Layoffs',
        data: [2982, 35441, 7982, 630, 5052, 196, 1830, 3222, 6970, 10824, 13593, 43168, 123874, 3876]
    }],
    chart: {
        height: 350,
        type: 'area',
        toolbar: {
            show: false,
        },
    },
    colors: ['#4f35a1', '#246dec'],
    dataLabels: {
        enabled: false,
    },
    stroke: {
        curve: 'smooth'
    },
    labels: ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4",
             "2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2"],
    markers: {
        size: 0
    },
    yaxis: [
        {
            title: {
                text: 'All Layoffs',
            },
        },
        {
            opposite: true,
            title: {
                text: 'STEM Layoffs',
            },
        },
    ],
    tooltip: {
        shared: true,
        intersect: false,
    }
};

var areaChart = new ApexCharts(document.querySelector("#area-chart"), areaChartOptions);
areaChart.render();



// Bar Chart for dropdown selection with title
// setup 
const data_bar = {
    labels: ['2018', '2019', '2020', '2021'],
    datasets: [{
      data: [
        {'id': '2018',
          'metadata': {'Business Analyst': 149000.0,
          'Data Scientist': 226503.87596899224,
          'Hardware Engineer': 238000.0,
          'Human Resources': 170000.0,
          'Marketing': 100000.0,
          'Product Designer': 210266.66666666666,
          'Product Manager': 276988.31775700935,
          'Recruiter': 170000.0,
          'Sales': 566500.0,
          'Software Engineer': 231179.47594323294,
          'Software Engineering Manager': 374254.23728813557,
          'Solution Architect': 244271.18644067796,
          'Technical Program Manager': 261500.0}},
        {'id': '2019',
          'metadata': {'Business Analyst': 116714.28571428571,
          'Data Scientist': 211633.64055299538,
          'Hardware Engineer': 221420.92457420923,
          'Human Resources': 200857.14285714287,
          'Marketing': 271428.5714285714,
          'Mechanical Engineer': 113875.0,
          'Product Designer': 213052.4193548387,
          'Product Manager': 263759.62815405044,
          'Recruiter': 179000.0,
          'Sales': 312555.55555555556,
          'Software Engineer': 219398.03498682004,
          'Software Engineering Manager': 372315.4269972452,
          'Solution Architect': 221489.08296943232,
          'Technical Program Manager': 180000.0}},
        {'id': '2020',
          'metadata': {'Business Analyst': 124845.10250569475,
          'Data Scientist': 202522.35772357724,
          'Hardware Engineer': 219348.83720930232,
          'Human Resources': 179453.125,
          'Management Consultant': 158977.64227642276,
          'Marketing': 200464.78873239437,
          'Mechanical Engineer': 155619.77186311787,
          'Product Designer': 208917.46031746033,
          'Product Manager': 249851.02834908283,
          'Recruiter': 152938.46153846153,
          'Sales': 208838.6075949367,
          'Software Engineer': 204269.1946308725,
          'Software Engineering Manager': 351456.12708018156,
          'Solution Architect': 212060.53811659192,
          'Technical Program Manager': 239265.98837209304}},
        {'id': '2021',
          'metadata': {'Business Analyst': 134720.18348623853,
          'Data Scientist': 198522.7934044617,
          'Hardware Engineer': 204625.0,
          'Human Resources': 176951.21951219512,
          'Management Consultant': 166675.61983471076,
          'Marketing': 196268.01152737753,
          'Mechanical Engineer': 163461.18721461188,
          'Product Designer': 203468.91651865008,
          'Product Manager': 258598.5576923077,
          'Recruiter': 158741.93548387097,
          'Sales': 215231.3432835821,
          'Software Engineer': 193447.18545260062,
          'Software Engineering Manager': 343458.6092715232,
          'Solution Architect': 204312.05673758866,
          'Technical Program Manager': 234952.1739130435}}]
        ,
      backgroundColor: [
        "#246dec",
        "#cc3c43",
        "#367952",
        "#4f35a1"
      ],
      borderWidth: 1
    }]
};

// config 
const config = {
    type: 'bar',
    data: data_bar,
    options: {
      parsing: {
        xAxisKey: 'id',
        yAxisKey: 'metadata.Business Analyst'
      },
      scales: {
        y: {
          beginAtZero: true
        }
      },
      plugins: {
        legend: {
            display: false
        }
      }
    }
};
//console.log(configpie)

// render init block
const myChart = new Chart(
    document.getElementById('myChart'),
    config
);

function changeTitle(){
    const title = document.getElementById('title').value;
    console.log(title);
    
    const output = `metadata.${title}`;
    console.log(output)

    myChart.config.options.parsing.yAxisKey = output;
    myChart.update();
}

// Instantly assign Chart.js version
const chartVersion = document.getElementById('chartVersion');
chartVersion.innerText = Chart.version;






////////////////////////////////////////////////
// Put your code here



