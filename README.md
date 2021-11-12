[<img width="200" alt="get in touch with Consensys Diligence" src="https://user-images.githubusercontent.com/2865694/56826101-91dcf380-685b-11e9-937c-af49c2510aa0.png">](https://diligence.consensys.net)<br/>
<sup>
[[  🌐  ](https://diligence.consensys.net)  [  📩  ](https://github.com/ConsenSys/vscode-solidity-doppelganger/blob/master/mailto:diligence@consensys.net)  [  🔥  ](https://consensys.github.io/diligence/)]
</sup><br/><br/>

<sub><b>This is a PoC for HackWek! - a Diligence internal 5-day Hackathon 🥷⚔️.<br>TLDR;</b> My plan was to have fun with tensorflow, RNN's, text-prediction, and connect this to solidity smart contracts 🙌. This is an excerpt of my journey.</sub>

# Hallucinate.sol

😵‍💫 A Recurrent Neural Network (RNN) hallucinating solidity source code.

* We train our model on samples from the https://github.com/tintinweb/smart-contract-sanctuary
* And then "hallucinate" new contracts


![image](https://user-images.githubusercontent.com/2865694/141459177-49d9d800-6da5-4736-b7f5-761546532160.png)

**Note**: train the model on https://colab.research.google.com/ as it is much faster than doing this locally.

## Interactive Playground

Copy the python notebook to your own collab/google drive and runit.

<sub>Hint: Google Collab → Runtime → Change Runtime Type: GPU</sub>

* 👉 [Tutorial 2 - load & hallucinate](https://drive.google.com/file/d/16vQX3SVxmqmkXfwWut38YPfcrRq4SlAE/view?usp=sharing)
* 👉 [Tutorial 1 - train & hallucinate](https://drive.google.com/file/d/13Z6Ak7UCUf6sMvCujeym2A6Bio8mTv66/view?usp=sharing)

## Contents


| Folder       | Description   |
| ------------ | ------------- |
| [solidity_model_text](./solidity_model_text/)    | contains a pre-trained model trained on 15mb solidity input, naive character based training, with sampling sequence length of 250 chars. The model has an `embedding_dimension` of `256` with `1024` `rnn_units`. It was trained for `15 epochs` on google collab (hw-accelleration: `GPU`) which took somewhere between 1-1.5 hrs. |
| [Tutorial 2: load & hallucinate](./tutorial_2_hallucinate_from_pretrained_model.ipynb)    | loads the pre-trained model from [./solidity_model_text/](./solidity_model_text/) and hallucinates more solidity. |
| [Tutorial 1: train & hallucinate](./tutorial_1_train_and_hallucinate_save_restore_continue_training.ipynb)        | is the code that downloads samples from https://github.com/tintinweb/smart-contract-sanctuary, creates the model, trains it, hallucinates some text, and then continues to show how to save/restore/re-train the model. |

* **Note**: The model can be exported for use with [tensorflow.js](https://www.tensorflow.org/js) so that it can be used with any javascript/web-front/backend. See [Tutorial 1](./tutorial_1_train_and_hallucinate_save_restore_continue_training.ipynb) for how to do this.
* **Note**: The model can also be used for non-solidity code. Just make sure to write your own `SolidityTrainer` class 🙌.


## Improvements

Of course, there's no way to explore everything in this 5-day HackWek period, but, here're a couple of thoughts on what to improve:

* vocabulary should be based on tokentype_text instead of chars. E.g. use `pygments` to lex `solidity` and map this as the vocabulary. This should give way higher quality output and allows the model to learn the source structure more efficiently.
* input cleanup should reliably remove all comments/pragmas/etc.
* loss function should reinforce training towards fuzzy-parseable code
* shuffle before downloading contract sources
* continuous learning. re-train with more sources (not only 15mb :D)

## Example

Copy the two tutorials to your google drive and run them.

**Input:**

```python
>>> print(trainingData.predict(['contract '], 3000))
```

**Output:**

```solidity
contract Ownable {
  address public owner;
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
  function Ownable() public {
    owner = msg.sender;
  }
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
}
contract Parminicinvition is Ownable {
    using SafeMath for uint256;
    enum State { Approve          = token.totalSupply();
      require(tokens >= summaryTokens.add(bonus));
        totalDailydested = totalEthInWei + msg.value;
        totalSoldTokens = token.totalSupply();
        emit Transfer(address(0), 0xCf49B9298aC4d4933a7D6984d89A49aDc84A6CA602BA513D872C3,21f36325D28718](0));
        totalSupply = totalSupply.mul(totalValue.add(soldSignedMap[tokensBough.mul(1)));
          restributedPluyRates[msg.sender] = true;
              nonStokenSupplyFinallow
        }
                if(opits[msg.sender].amount <= totalSupply)) ether;
			}
		assignOpe(address(this).balance, weiAmount);
		require(canTra_secrecover(_approved) >= rNo(_reward, _weight, _amount);
	    totalAmount = totalAmount.add(_amount);
        Transfer(_addr, msg.sender, amount);
    }
...
```


## Credits

Based on the [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)
