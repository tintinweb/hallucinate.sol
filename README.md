[<img width="200" alt="get in touch with Consensys Diligence" src="https://user-images.githubusercontent.com/2865694/56826101-91dcf380-685b-11e9-937c-af49c2510aa0.png">](https://diligence.consensys.net)<br/>
<sup>
[[  🌐  ](https://diligence.consensys.net)  [  📩  ](https://github.com/ConsenSys/vscode-solidity-doppelganger/blob/master/mailto:diligence@consensys.net)  [  🔥  ](https://consensys.github.io/diligence/)]
</sup><br/><br/>


# Hallucinate.sol

😵‍💫 A Recurrent Neural Network (RNN) is hallucinating solidity source code.

* We train our model on samples from the https://github.com/tintinweb/smart-contract-sanctuary
* And then "hallucinate" new contracts


## Contents

* `./solidity_model_text/` - contains a pre-trained keras model trained on 15mb solidity input, character based training (pretty dumb), with sampling sequence length of 250 chars. The model has an `embedding_dimension` of `256` with `1024` `rnn_units`. It was trained for `15 epochs` on google collab (hw-accelleration: `GPU`) which took somewhere  between 1-1.5 hrs.
* [./tutorial_2_hallucinate_from_pretrained_model.ipynb](./tutorial_2_hallucinate_from_pretrained_model.ipynb) - loads the pre-trained model from `./solidity_model_text` to hallucinate more solidity.
* [./tutorial_1_train_and_hallucinate_save_restore_continue_training.ipynb](./tutorial_1_train_and_hallucinate_save_restore_continue_training.ipynb) - is the code that downloads samples from https://github.com/tintinweb/smart-contract-sanctuary, creates the model, trains it, hallucinates some text, and then continues to show how to save/restore/re-train the model. Ultimately, the model is exported and converted to [tensorflow.js](https://www.tensorflow.org/js) conform format so that it can be used with any javascript/web-front/backend.

## Demo

Copy the two tutorials to your google drive and run them.

```python
print(trainingData.predict(['contract '], 3000))
```

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
    function checkRollUnerad(address _caller) external {
        require(activeTransieronase / _values[i] >= eth[i] && _patent.beneficiary != address(0));
        require(_amountOfTokens <= tokenBalanceLedger_[_customerAddress]);
        uint256 _tokens = _amountOfTokens;
        uint256 _ethereum = tokensToEthereum_(_tokens, _taxedEthereum);
        uint256 _divTokens = getToken().balanceOf(_owner);
        require(_tokenAddress >= _value);
        allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    }
    function multiSigWallet(address[] _KeceiveAddress(address _whenAddress) onlyOwner public {
        stats[msg.sender][_i] = _getUnlockedAmt;
    }
    function enableKYC() external onlyOwner beforeURe {
        uint256 lastIndex = 0;
        address _tokenAddress indexe  msg.sender;
        uint128 accountKey = uint128(_affidiets, userTrustedContribution);
        assignUpgredateTokens(_address, _beneficiever, _tokens);
    }
    function setRestrictAddressLass() onlyOwnerships onlyWhitelisted r
      uint256 _remainingTokens = 
         (_tokenPriceInitial * tokenPriceIncremental_ * (_tokenSupply/1e18);
            uint depositAmount = deadline );
        }
        isReleaseAgent = addr;
    }
    function paymentSuspendedPaymentsale(uint256[] _users, uint256[] _scriptLogoAssess) public payable returns(uint256[12] _eth) {
        k.cards[0x343C2593c3216d33433D43fBBF6cF7 = txp1 * item_iXCalculated[_userNumber].enity_inceent, 0.0000001 ether, other.umLock());
    }
    function buyTokens(address _address) external onlyWhileOter{
        uint256 inReadyToBivs;
        totalSupply = totalSupply.sub(_amount);
        DistrustedContributionTime = _block.pixe 
```


## Credits

Based on the [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)